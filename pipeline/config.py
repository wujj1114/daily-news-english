"""組態載入、環境變數讀取與模型可用性檢查。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "show.yaml"
TAIPEI_TZ = ZoneInfo("Asia/Taipei")


@dataclass(frozen=True)
class AppSettings:
    """環境變數設定（不含敏感字串於 log）。"""

    gemini_api_key: str
    gemini_script_model: str
    gemini_tts_model: str
    max_tts_chars_per_run: int


def load_settings(env_path: Path | None = None) -> AppSettings:
    """載入 .env 與環境變數。"""
    load_dotenv(dotenv_path=env_path or (PROJECT_ROOT / ".env"), override=True)
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key and "GOOGLE_API_KEY" in os.environ:
        # 避免 google-genai SDK 優先讀取本機舊有的 GOOGLE_API_KEY 而忽略 .env 的 GEMINI_API_KEY
        os.environ.pop("GOOGLE_API_KEY", None)
    return AppSettings(
        gemini_api_key=gemini_key,
        gemini_script_model=os.getenv("GEMINI_SCRIPT_MODEL", "gemini-2.5-flash").strip(),
        gemini_tts_model=os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts").strip(),
        max_tts_chars_per_run=int(os.getenv("MAX_TTS_CHARS_PER_RUN", "25000")),
    )


def load_show_config(config_path: Path | None = None) -> dict[str, Any]:
    """讀取 config/show.yaml。"""
    path = config_path or DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"組態檔格式錯誤：{path}")
    return data


def parse_episode_date(date_arg: str) -> str:
    """將 'today' 或 'YYYY-MM-DD' 正規化為 'YYYY-MM-DD'（臺北時區）。"""
    if date_arg.lower() == "today":
        return datetime.now(TAIPEI_TZ).date().isoformat()
    parsed = date.fromisoformat(date_arg)
    return parsed.isoformat()


def get_episode_dir(episode_date: str, base_dir: Path | None = None) -> Path:
    """取得並建立指定日期的產物目錄。"""
    root = base_dir or PROJECT_ROOT
    ep_dir = root / "episodes" / episode_date
    ep_dir.mkdir(parents=True, exist_ok=True)
    return ep_dir


def check_models_available(client: Any, model_names: list[str]) -> None:
    """啟動時檢查設定的 Gemini 模型是否可於 API 取得。"""
    try:
        available = {
            m.name.removeprefix("models/")
            for m in client.models.list()
            if getattr(m, "name", None)
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("無法列出 Gemini 模型清單（將直接嘗試呼叫）：%s", exc)
        return

    for name in model_names:
        clean = name.removeprefix("models/")
        if clean not in available:
            raise RuntimeError(
                f"環境變數指定的模型 '{name}' 不在目前可用的 Gemini 模型清單中，"
                "請檢查 GEMINI_SCRIPT_MODEL / GEMINI_TTS_MODEL 設定。"
            )
