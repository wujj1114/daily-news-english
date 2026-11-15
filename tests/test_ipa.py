"""CMUdict 轉 IPA 音標測試。"""

import logging

from pipeline.ipa import lookup_ipa


def test_lookup_ipa_known_word() -> None:
    ipa = lookup_ipa("unchanged")
    assert ipa.startswith("/") and ipa.endswith("/")
    assert "tʃ" in ipa


def test_lookup_ipa_unknown_word_returns_empty_and_warns(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        ipa = lookup_ipa("xyzzynotaword")
    assert ipa == ""
    assert "CMUdict 查無單字音標" in caplog.text
