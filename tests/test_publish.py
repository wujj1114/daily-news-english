"""Podcast RSS 產生與同日重跑冪等性測試。"""

import xml.etree.ElementTree as ET
from pathlib import Path

from pipeline.config import load_show_config
from pipeline.publish import ITUNES_NS, update_rss_feed
from tests.test_schema import _sample_episode


def test_update_rss_feed_idempotent_and_valid_xml(tmp_path: Path) -> None:
    show_cfg = load_show_config()
    feed_path = tmp_path / "docs" / "feed.xml"
    ep = _sample_episode()

    update_rss_feed(
        feed_path=feed_path,
        episode=ep,
        show_cfg=show_cfg,
        audio_url="https://github.com/wujj1114/daily-news-english/releases/download/ep-2026-11-15/episode.mp3",
        audio_bytes=5_000_000,
        duration_sec=423.9,
    )
    # 同日重跑應覆蓋既有 item，不得重複新增
    update_rss_feed(
        feed_path=feed_path,
        episode=ep,
        show_cfg=show_cfg,
        audio_url="https://github.com/wujj1114/daily-news-english/releases/download/ep-2026-11-15/episode.mp3",
        audio_bytes=5_100_000,
        duration_sec=425.0,
    )

    tree = ET.parse(feed_path)
    channel = tree.getroot().find("channel")
    assert channel is not None
    items = channel.findall("item")
    assert len(items) == 1
    enclosure = items[0].find("enclosure")
    assert enclosure is not None
    assert enclosure.attrib["length"] == "5100000"
    assert enclosure.attrib["type"] == "audio/mpeg"
    duration_el = items[0].find(f"{{{ITUNES_NS}}}duration")
    assert duration_el is not None
    assert duration_el.text == "00:07:05"
