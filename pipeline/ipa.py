"""CMUdict 轉 IPA 音標查詢工具。

嚴格依賴字典資料（cmudict）轉換為 IPA；查無時回傳空字串並記錄警告，
不由大型語言模型自行生成音標。
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import cmudict

logger = logging.getLogger(__name__)

# ARPAbet 音素對應 IPA（美式英語標準轉換）
_ARPABET_TO_IPA: dict[str, str] = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}


@lru_cache(maxsize=1)
def _load_cmudict() -> dict[str, list[list[str]]]:
    return cmudict.dict()


def arpabet_to_ipa(phones: list[str]) -> str:
    """將一組 ARPAbet 音素序列轉為帶重音符號的 IPA 字串（含斜線）。"""
    parts: list[str] = []
    for token in phones:
        match = re.fullmatch(r"([A-Z]+)([012]?)", token)
        if not match:
            continue
        base, stress = match.groups()
        ipa_sym = _ARPABET_TO_IPA.get(base)
        if not ipa_sym:
            continue
        # 非重音的 AH / ER 常用弱化形式 ə / ɚ
        if stress == "0" and base == "AH":
            ipa_sym = "ə"
        elif stress == "0" and base == "ER":
            ipa_sym = "ɚ"

        if stress == "1":
            parts.append("ˈ" + ipa_sym)
        elif stress == "2":
            parts.append("ˌ" + ipa_sym)
        else:
            parts.append(ipa_sym)
    return f"/{''.join(parts)}/" if parts else ""


def lookup_ipa(word: str) -> str:
    """查詢英文單字或片語的 IPA 音標；查無時記錄警告並回傳空字串。"""
    cleaned = word.strip().lower()
    if not cleaned:
        return ""

    cmu = _load_cmudict()

    # 支援多字片語或連字號複合字逐字查詢
    tokens = re.split(r"[\s\-]+", cleaned)
    ipa_tokens: list[str] = []
    for tok in tokens:
        pure = re.sub(r"^[^a-z']+|[^a-z']+$", "", tok)
        if not pure:
            continue
        entries = cmu.get(pure)
        if not entries:
            logger.warning("CMUdict 查無單字音標，將留空：%s", word)
            return ""
        ipa = arpabet_to_ipa(entries[0])
        if not ipa:
            logger.warning("ARPAbet 轉換 IPA 失敗，將留空：%s", word)
            return ""
        ipa_tokens.append(ipa.strip("/"))

    if not ipa_tokens:
        return ""
    return f"/{' '.join(ipa_tokens)}/"
