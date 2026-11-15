"""episode.json Schema 與規則驗證測試。"""

from copy import deepcopy

from pipeline.schema import validate_episode


def _sample_episode() -> dict:
    sentences = []
    vocab_words = [
        "economy", "inflation", "interest", "policy",
        "tourism", "airline", "visitor", "schedule",
        "software", "network",
    ]
    idx = 0
    stories = []
    for s in range(3):
        s_sents = []
        for i in range(4):
            v_list = []
            if idx < len(vocab_words):
                v_list.append({
                    "word": vocab_words[idx],
                    "ipa": "/ˈtest/",
                    "pos": "n.",
                    "zh": "測試詞",
                })
                idx += 1
            s_sents.append({
                "id": f"s{s * 4 + i + 1:02d}",
                "en": "Short news sentence for students to learn English clearly.",
                "zh": "適合學生學習英文的簡短新聞句子。",
                "vocab": v_list,
            })
        stories.append({
            "topic": f"領域{s + 1}",
            "headline_en": f"Headline {s + 1}",
            "headline_zh": f"新聞標題 {s + 1}",
            "sources": ["https://example.com/a", "https://example.org/b"],
            "sentences": s_sents,
        })
    return {
        "date": "2026-11-15",
        "title_en": "NEWS IN 10 — Nov 15",
        "title_zh": "NEWS IN 10 每日新聞英語｜11 月 15 日",
        "stories": stories,
    }


def test_valid_episode_passes() -> None:
    assert validate_episode(_sample_episode()) == []


def test_sentence_over_22_words_fails() -> None:
    ep = _sample_episode()
    ep["stories"][0]["sentences"][0]["en"] = " ".join(["word"] * 23) + "."
    errors = validate_episode(ep)
    assert any("超過上限 22 字" in e for e in errors)


def test_duplicate_vocab_fails() -> None:
    ep = _sample_episode()
    ep["stories"][1]["sentences"][0]["vocab"][0]["word"] = "economy"
    errors = validate_episode(ep)
    assert any("生難字重複出現" in e for e in errors)


def test_single_source_fails() -> None:
    ep = _sample_episode()
    ep["stories"][0]["sources"] = ["https://example.com/only-one"]
    errors = validate_episode(ep)
    assert len(errors) > 0
