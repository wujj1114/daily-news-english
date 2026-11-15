"""TTS 合成、拼讀格式與內容雜湊快取測試。"""

import json
from pathlib import Path

from mutagen.id3 import ID3

from pipeline.config import AppSettings, load_show_config
from pipeline.tts import TTSEngine, _make_silence_wav, build_tts_plan, format_spelling_prompt, synthesize_episode
from tests.test_schema import _sample_episode


class FakeTTSEngine(TTSEngine):
    """測試用假 TTS 引擎，計算實際合成呼叫次數。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, float]] = []

    def synthesize(self, text: str, lang: str, voice: str, rate: float = 1.0) -> bytes:
        self.calls.append((text, lang, voice, rate))
        return _make_silence_wav(0.2, sample_rate=24000)


def test_format_spelling_prompt() -> None:
    assert format_spelling_prompt("unchanged") == "U, N, C, H, A, N, G, E, D"


def test_tts_plan_en_before_zh() -> None:
    show_cfg = load_show_config()
    ep = _sample_episode()
    plan = build_tts_plan(ep, show_cfg)
    speech_items = [p for p in plan if p["kind"] == "speech"]
    en_idx = next(i for i, p in enumerate(speech_items) if p["type"] == "en")
    assert speech_items[en_idx + 1]["type"] == "zh"
    assert speech_items[en_idx + 1]["sentence_id"] == speech_items[en_idx]["sentence_id"]


def test_synthesize_episode_and_cache_hit(tmp_path: Path) -> None:
    show_cfg = load_show_config()
    ep_dir = tmp_path / "episodes" / "2026-11-15"
    ep_dir.mkdir(parents=True)
    ep_dir.joinpath("episode.json").write_text(
        json.dumps(_sample_episode(), ensure_ascii=False), encoding="utf-8"
    )
    cache_dir = tmp_path / "cache"
    settings = AppSettings(
        gemini_api_key="dummy",
        gemini_script_model="gemini-2.5-flash",
        gemini_tts_model="gemini-2.5-flash-preview-tts",
        max_tts_chars_per_run=50000,
    )

    engine1 = FakeTTSEngine()
    meta1 = synthesize_episode(ep_dir, settings, show_cfg, engine=engine1, cache_dir=cache_dir)
    assert (ep_dir / "episode.mp3").exists()
    assert (ep_dir / "segments.json").exists()
    assert meta1["cache_misses"] > 0
    assert len(engine1.calls) == meta1["cache_misses"]

    # 驗證 ID3 標籤
    tags = ID3(str(ep_dir / "episode.mp3"))
    assert "NEWS IN 10" in str(tags.get("TALB"))

    # 第二次重跑：必須全部命中快取，無新增 TTS 呼叫
    engine2 = FakeTTSEngine()
    meta2 = synthesize_episode(ep_dir, settings, show_cfg, engine=engine2, cache_dir=cache_dir)
    assert meta2["cache_misses"] == 0
    assert meta2["cache_hits"] == len(meta2["segments"])
    assert len(engine2.calls) == 0
