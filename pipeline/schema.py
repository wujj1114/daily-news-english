"""episode.json 與 timeline.json 的 JSON Schema 與語意驗證。"""

from __future__ import annotations

import re
from typing import Any

import jsonschema

EPISODE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["date", "title_en", "title_zh", "stories"],
    "additionalProperties": True,
    "properties": {
        "date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "title_en": {"type": "string", "minLength": 1},
        "title_zh": {"type": "string", "minLength": 1},
        "stories": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["headline_en", "headline_zh", "sources", "sentences"],
                "additionalProperties": True,
                "properties": {
                    "topic": {"type": "string"},
                    "headline_en": {"type": "string", "minLength": 1},
                    "headline_zh": {"type": "string", "minLength": 1},
                    "sources": {
                        "type": "array",
                        "minItems": 2,
                        "items": {"type": "string", "pattern": r"^https?://"},
                    },
                    "sentences": {
                        "type": "array",
                        "minItems": 4,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "required": ["id", "en", "zh", "vocab"],
                            "additionalProperties": True,
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "en": {"type": "string", "minLength": 1},
                                "zh": {"type": "string", "minLength": 1},
                                "vocab": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": 2,
                                    "items": {
                                        "type": "object",
                                        "required": ["word", "ipa", "pos", "zh"],
                                        "additionalProperties": True,
                                        "properties": {
                                            "word": {"type": "string", "minLength": 1},
                                            "ipa": {"type": "string"},
                                            "pos": {"type": "string", "minLength": 1},
                                            "zh": {"type": "string", "minLength": 1},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def count_en_words(sentence: str) -> int:
    """計算英文句子字數。"""
    return len([w for w in sentence.strip().split() if w])


def validate_episode(episode: dict[str, Any], content_cfg: dict[str, Any] | None = None) -> list[str]:
    """驗證 episode.json 是否符合 Schema 與節目規則，回傳錯誤訊息清單。"""
    errors: list[str] = []
    validator = jsonschema.Draft202012Validator(EPISODE_SCHEMA)
    for err in sorted(validator.iter_errors(episode), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        errors.append(f"Schema 錯誤 [{loc}]: {err.message}")

    if errors:
        return errors

    cfg = content_cfg or {}
    max_words = int(cfg.get("max_en_words_per_sentence", 22))
    vocab_min = int(cfg.get("vocab_min_total", 8))
    vocab_max = int(cfg.get("vocab_max_total", 12))

    seen_vocab: set[str] = set()
    total_vocab = 0
    for s_idx, story in enumerate(episode["stories"], start=1):
        for sent in story["sentences"]:
            if re.search(r"[\u4e00-\u9fff]", sent["en"]):
                errors.append(f"第 {s_idx} 則新聞句子 {sent['id']} 英文欄含有中文字元")
            word_count = count_en_words(sent["en"])
            if word_count > max_words:
                errors.append(
                    f"第 {s_idx} 則新聞句子 {sent['id']} 英文長度為 {word_count} 字，超過上限 {max_words} 字"
                )
            for v in sent["vocab"]:
                key = v["word"].strip().lower()
                if key in seen_vocab:
                    errors.append(f"生難字重複出現：{v['word']}")
                seen_vocab.add(key)
                total_vocab += 1

    if not (vocab_min <= total_vocab <= vocab_max):
        errors.append(f"全集生難字總數為 {total_vocab}，應介於 {vocab_min}–{vocab_max} 個之間")

    return errors
