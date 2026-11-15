"""階段 1：新聞搜尋、雙重查核與腳本生成（產出 episode.json）。"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from pipeline.config import AppSettings, TAIPEI_TZ, check_models_available
from pipeline.ipa import lookup_ipa
from pipeline.schema import validate_episode

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict[str, Any]:
    """從模型回覆中擷取 JSON 物件（支援 ```json 區塊）。"""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    candidate = fence_match.group(1) if fence_match else stripped
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("模型回覆中找不到有效的 JSON 物件")
    return json.loads(candidate[start : end + 1])


def _call_with_backoff(fn: Any, max_attempts: int = 3, base_delay: float = 2.0) -> Any:
    """指數退避重試包裝器（最多 3 次）。"""
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("Gemini API 呼叫失敗（第 %d/%d 次），%.1f 秒後重試：%s", attempt, max_attempts, delay, exc)
            time.sleep(delay)
    raise RuntimeError(f"Gemini API 連續 {max_attempts} 次失敗：{last_exc}") from last_exc


def _normalize_episode(episode: dict[str, Any]) -> None:
    """補齊可省略的預設欄位、移除多餘 Markdown 標記並以 CMUdict 查詢所有生難字 IPA 音標。"""
    for story in episode.get("stories", []):
        for sent in story.get("sentences", []):
            if isinstance(sent.get("en"), str):
                sent["en"] = sent["en"].replace("**", "").strip()
            if isinstance(sent.get("zh"), str):
                sent["zh"] = sent["zh"].replace("**", "").strip()
            if "vocab" not in sent or sent["vocab"] is None:
                sent["vocab"] = []
            for vocab in sent["vocab"]:
                vocab["ipa"] = lookup_ipa(vocab.get("word", ""))


def _build_draft_prompt(
    episode_date: str,
    show_cfg: dict[str, Any],
    sentences_hint: str = "4 到 5 句",
    previous_errors: list[str] | None = None,
) -> str:
    show = show_cfg["show"]
    content = show_cfg["content"]
    topics = "、".join(content.get("topics", ["臺灣", "AI 發展", "生活", "旅遊", "財經"]))
    avoid = "、".join(content.get("avoid_topics", ["血腥暴力細節", "未經證實的謠言", "主觀政治評論"]))
    dt = datetime.strptime(episode_date, "%Y-%m-%d").replace(tzinfo=TAIPEI_TZ)
    date_en = dt.strftime("%b %d").replace(" 0", " ")
    date_zh = f"{dt.month} 月 {dt.day} 日"

    error_feedback = ""
    if previous_errors:
        joined = "\n".join(f"- {e}" for e in previous_errors)
        error_feedback = f"\n\n【前次驗證失敗，請務必修正以下問題】\n{joined}\n"

    return f"""請使用 Google 搜尋查詢過去 24 小時（基準日期：{episode_date}，臺北時間）的真實時事新聞，為英語教學節目「{show['name_en']}」撰寫一集節目腳本。

【選題與內容規範】
1. 從以下領域中選出 3 則「不同領域」的重大或有趣新聞：{topics}。
2. 嚴格避開：{avoid}；政治新聞僅陳述事實，不加主觀評論。
3. 僅取新聞事實，以自身文字重新撰寫，不得直接翻譯或逐字抄襲單一報導文句。
4. 每則新聞必須至少有 2 個不同且真實存在的 `https://` 新聞來源網址（列於 `sources`）。
5. 所有中文內容（`headline_zh`、`zh`）必須使用「臺灣繁體中文」與「臺灣慣用語」（例如：晶片、人工智慧、軟體、伺服器、捷運等，勿用中國大陸用語）。

【句構與生難字規範】
1. 每則新聞共 {sentences_hint}（`sentences` 長度必須為 4 或 5）。
2. 每句英文 `en` 長度嚴格限制在 {content.get('max_en_words_per_sentence', 22)} 個英文單字以內，語氣自然清楚；超過請拆句。
3. 每句搭配對應的中文翻譯/說明 `zh`。
4. 生難字標準：{content.get('vocab_level', 'CEFR A2-B1（臺灣國三剛升高一程度）')} 的實用單字（每句挑選 0 到 2 個）。
5. 整集 3 則新聞合計生難字總數必須嚴格介於 {content.get('vocab_min_total', 8)} 到 {content.get('vocab_max_total', 12)} 個之間，且整集不可重複出現同一個 `word`。
6. `vocab` 的 `ipa` 欄位請先留空字串 `""`（後續由系統字典自動填入）。
7. `vocab` 的 `pos` 請使用簡寫如 `n.`, `v.`, `adj.`, `adv.`, `phr.`。
{error_feedback}
請只輸出純 JSON（可包在 ```json ... ``` 中，不要有其他說明文字），格式如下：
{{
  "date": "{episode_date}",
  "title_en": "{show['name_en']} — {date_en}",
  "title_zh": "{show['name_zh']}｜{date_zh}",
  "stories": [
    {{
      "topic": "領域名稱",
      "headline_en": "英文標題",
      "headline_zh": "臺灣繁體中文標題",
      "sources": ["https://...", "https://..."],
      "sentences": [
        {{
          "id": "s01",
          "en": "Short English sentence within 22 words.",
          "zh": "對應的臺灣繁體中文句子。",
          "vocab": [
            {{
              "word": "example",
              "ipa": "",
              "pos": "n.",
              "zh": "例子；範例"
            }}
          ]
        }}
      ]
    }}
  ]
}}"""


def _build_verify_prompt(draft: dict[str, Any], show_cfg: dict[str, Any]) -> str:
    content = show_cfg["content"]
    draft_json = json.dumps(draft, ensure_ascii=False, indent=2)
    return f"""你是一位嚴謹的新聞事實查核員與臺灣繁體中文編輯。請使用 Google 搜尋查核以下 JSON 腳本：

1. **事實查核**：逐句比對 `sources` 與公開新聞資料，移除或修正任何無法佐證的數字、時間或陳述；確認每則新聞至少有 2 個真實有效的 `https://` 來源網址。
2. **臺灣慣用語檢查**：將所有中國大陸用語修正為臺灣繁體中文慣用語（例如：視頻→影片、信息→資訊、軟件→軟體、人工智能→人工智慧、芯片→晶片、網絡→網路等）。
3. **規格檢查**：
   - 共 3 則新聞，每則 `sentences` 為 4–5 句。
   - 每句 `en` 不超過 {content.get('max_en_words_per_sentence', 22)} 個英文單字。
   - 整集 `vocab` 總數介於 {content.get('vocab_min_total', 8)}–{content.get('vocab_max_total', 12)} 個，不可重複，難度適合 {content.get('vocab_level', 'CEFR A2-B1')}。
   - `ipa` 保持為 `""`，`id` 依序為 `s01`、`s02`...。

待查核腳本：
```json
{draft_json}
```

請只輸出查核修正後的完整 JSON（不要有其他說明文字）。"""


def generate_episode_script(
    episode_date: str,
    episode_dir: Path,
    settings: AppSettings,
    show_cfg: dict[str, Any],
    sentences_hint: str = "4 到 5 句",
    client: Any | None = None,
) -> dict[str, Any]:
    """執行階段 1：搜尋新聞、二次查核、字典查 IPA、驗證並存為 episode.json。"""
    if not settings.gemini_api_key and client is None:
        raise RuntimeError("未設定 GEMINI_API_KEY，請先於 .env 中填入有效的金鑰。")

    if client is None:
        client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(api_version="v1beta"),
        )
        check_models_available(client, [settings.gemini_script_model])

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    gen_config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.2,
    )

    content_cfg = show_cfg.get("content", {})
    max_validation_retries = 2
    last_errors: list[str] = []

    for attempt in range(1, max_validation_retries + 2):
        logger.info("階段 1：開始產生 %s 新聞腳本（第 %d 次嘗試）...", episode_date, attempt)
        draft_prompt = _build_draft_prompt(
            episode_date=episode_date,
            show_cfg=show_cfg,
            sentences_hint=sentences_hint,
            previous_errors=last_errors or None,
        )

        try:
            draft_resp = _call_with_backoff(
                lambda: client.models.generate_content(
                    model=settings.gemini_script_model,
                    contents=draft_prompt,
                    config=gen_config,
                )
            )
            draft_data = _extract_json(draft_resp.text or "")

            logger.info("階段 1：進行第二次呼叫事實查核與臺灣用語校正...")
            verify_prompt = _build_verify_prompt(draft_data, show_cfg)
            verify_resp = _call_with_backoff(
                lambda: client.models.generate_content(
                    model=settings.gemini_script_model,
                    contents=verify_prompt,
                    config=gen_config,
                )
            )
            verified_data = _extract_json(verify_resp.text or "")
        except (ValueError, json.JSONDecodeError) as exc:
            last_errors = [f"JSON 解析失敗：{exc}"]
            logger.warning("腳本 JSON 解析失敗（第 %d 次）：%s", attempt, exc)
            continue

        _normalize_episode(verified_data)
        errors = validate_episode(verified_data, content_cfg=content_cfg)
        if not errors:
            out_path = episode_dir / "episode.json"
            out_path.write_text(json.dumps(verified_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("階段 1 完成：已寫入 %s", out_path)
            return verified_data

        last_errors = errors
        logger.warning("腳本驗證失敗（第 %d 次）：%s", attempt, "; ".join(errors))

    raise RuntimeError(
        f"腳本驗證連續失敗（已重試 {max_validation_retries} 次），終止當日流程："
        + " | ".join(last_errors)
    )
