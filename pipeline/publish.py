"""階段 5（Podcast 部分）：上傳音訊至 GitHub Releases 並更新 GitHub Pages RSS (docs/feed.xml)。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.dom import minidom

from mutagen.mp3 import MP3

from pipeline.config import PROJECT_ROOT, TAIPEI_TZ

logger = logging.getLogger(__name__)

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
ATOM_NS = "http://www.w3.org/2005/Atom"

ET.register_namespace("itunes", ITUNES_NS)
ET.register_namespace("atom", ATOM_NS)


def _format_duration_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_episode_description(episode: dict[str, Any]) -> str:
    """組合節目說明欄文字（含新聞標題、逐句文字、生難字與來源連結）。"""
    lines: list[str] = [episode.get("title_zh", ""), ""]
    for idx, story in enumerate(episode.get("stories", []), start=1):
        lines.append(f"【新聞 {idx}】{story.get('headline_zh', '')}")
        lines.append(story.get("headline_en", ""))
        for sent in story.get("sentences", []):
            lines.append(f"• {sent.get('en', '')}")
            lines.append(f"  {sent.get('zh', '')}")
            for v in sent.get("vocab", []):
                ipa = f" {v['ipa']}" if v.get("ipa") else ""
                lines.append(f"  - {v['word']}{ipa} ({v.get('pos', '')}) {v.get('zh', '')}")
        lines.append("來源：")
        for src in story.get("sources", []):
            lines.append(f"  {src}")
        lines.append("")
    return "\n".join(lines).strip()


def upload_mp3_to_github_release(mp3_path: Path, episode_date: str, repo: str | None = None) -> str:
    """將 MP3 上傳至 GitHub Releases（以日期為唯一 tag，可覆寫重跑）並回傳公開下載網址。"""
    if not mp3_path.exists():
        raise FileNotFoundError(f"找不到音訊檔：{mp3_path}")

    tag = f"ep-{episode_date}"
    repo_arg = repo or os.getenv("GITHUB_REPOSITORY", "").strip()
    if not repo_arg:
        res = subprocess.run(
            ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_arg = res.stdout.strip()

    # 檢查 Release 是否已存在，不存在則建立
    view_res = subprocess.run(
        ["gh", "release", "view", tag, "--repo", repo_arg],
        capture_output=True,
        text=True,
    )
    if view_res.returncode != 0:
        subprocess.run(
            [
                "gh",
                "release",
                "create",
                tag,
                str(mp3_path),
                "--repo",
                repo_arg,
                "--title",
                f"NEWS IN 10 ({episode_date})",
                "--notes",
                f"Daily episode audio for {episode_date}",
            ],
            check=True,
        )
    else:
        subprocess.run(
            ["gh", "release", "upload", tag, str(mp3_path), "--repo", repo_arg, "--clobber"],
            check=True,
        )

    return f"https://github.com/{repo_arg}/releases/download/{tag}/{mp3_path.name}"


def update_rss_feed(
    feed_path: Path,
    episode: dict[str, Any],
    show_cfg: dict[str, Any],
    audio_url: str,
    audio_bytes: int,
    duration_sec: float,
) -> Path:
    """更新 docs/feed.xml；以日期為唯一鍵（guid），同日重跑時覆蓋原項目不重複新增。"""
    feed_path.parent.mkdir(parents=True, exist_ok=True)
    show = show_cfg["show"]
    podcast = show_cfg["podcast"]
    ep_date = episode["date"]
    guid_value = f"news-in-10-{ep_date}"

    rss = ET.Element("rss", {"version": "2.0"})
    channel = ET.SubElement(rss, "channel")

    ET.SubElement(channel, "title").text = show["name_zh"]
    ET.SubElement(channel, "link").text = podcast["site_url"]
    ET.SubElement(channel, "language").text = show.get("language", "zh-tw")
    ET.SubElement(channel, "description").text = show["description"]
    ET.SubElement(
        channel,
        f"{{{ATOM_NS}}}link",
        {"href": podcast["feed_url"], "rel": "self", "type": "application/rss+xml"},
    )
    ET.SubElement(channel, f"{{{ITUNES_NS}}}author").text = show.get("author", show["name_en"])
    ET.SubElement(channel, f"{{{ITUNES_NS}}}explicit").text = "true" if podcast.get("explicit") else "false"
    ET.SubElement(channel, f"{{{ITUNES_NS}}}image", {"href": podcast["cover_url"]})

    cat_el = ET.SubElement(channel, f"{{{ITUNES_NS}}}category", {"text": podcast.get("category", "Education")})
    if podcast.get("subcategory"):
        ET.SubElement(cat_el, f"{{{ITUNES_NS}}}category", {"text": podcast["subcategory"]})

    if podcast.get("email"):
        owner_el = ET.SubElement(channel, f"{{{ITUNES_NS}}}owner")
        ET.SubElement(owner_el, f"{{{ITUNES_NS}}}name").text = show.get("author", show["name_en"])
        ET.SubElement(owner_el, f"{{{ITUNES_NS}}}email").text = podcast["email"]

    # 讀取既有 <item> 並過濾掉同日期的舊項目（確保冪等）
    existing_items: list[ET.Element] = []
    if feed_path.exists():
        old_tree = ET.parse(feed_path)
        old_channel = old_tree.getroot().find("channel")
        if old_channel is not None:
            for item in old_channel.findall("item"):
                g = item.find("guid")
                if g is not None and g.text == guid_value:
                    continue
                existing_items.append(item)

    # 建立本集 <item>
    new_item = ET.Element("item")
    ET.SubElement(new_item, "title").text = episode["title_zh"]
    ET.SubElement(new_item, "description").text = build_episode_description(episode)
    ET.SubElement(new_item, "guid", {"isPermaLink": "false"}).text = guid_value

    pub_dt = datetime.strptime(ep_date, "%Y-%m-%d").replace(hour=6, minute=0, tzinfo=TAIPEI_TZ)
    ET.SubElement(new_item, "pubDate").text = format_datetime(pub_dt)
    ET.SubElement(
        new_item,
        "enclosure",
        {"url": audio_url, "length": str(audio_bytes), "type": "audio/mpeg"},
    )
    ET.SubElement(new_item, f"{{{ITUNES_NS}}}duration").text = _format_duration_hms(duration_sec)
    ET.SubElement(new_item, f"{{{ITUNES_NS}}}explicit").text = "true" if podcast.get("explicit") else "false"

    # 新集數排在最前面
    all_items = [new_item] + existing_items
    for item in all_items:
        channel.append(item)

    raw_xml = ET.tostring(rss, encoding="utf-8")
    pretty_xml = minidom.parseString(raw_xml).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")
    # 移除 minidom 產生的多餘空白行
    cleaned_lines = [line for line in pretty_xml.splitlines() if line.strip()]
    feed_path.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    return feed_path


def publish_podcast(
    episode_dir: Path,
    show_cfg: dict[str, Any],
    docs_dir: Path | None = None,
    assets_dir: Path | None = None,
) -> Path:
    """執行 Podcast 發布：上傳 MP3、複製封面圖並更新 docs/feed.xml。"""
    ep_path = episode_dir / "episode.json"
    mp3_path = episode_dir / "episode.mp3"
    if not ep_path.exists() or not mp3_path.exists():
        raise FileNotFoundError(f"缺少 episode.json 或 episode.mp3：{episode_dir}")

    episode = json.loads(ep_path.read_text(encoding="utf-8"))
    audio_bytes = mp3_path.stat().st_size
    duration_sec = MP3(str(mp3_path)).info.length

    resolved_docs = docs_dir or (PROJECT_ROOT / "docs")
    resolved_assets = assets_dir or (PROJECT_ROOT / "assets")
    cover_src = resolved_assets / "cover.jpg"
    if cover_src.exists():
        resolved_docs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cover_src, resolved_docs / "cover.jpg")

    audio_url = upload_mp3_to_github_release(mp3_path=mp3_path, episode_date=episode["date"])
    feed_path = update_rss_feed(
        feed_path=resolved_docs / "feed.xml",
        episode=episode,
        show_cfg=show_cfg,
        audio_url=audio_url,
        audio_bytes=audio_bytes,
        duration_sec=duration_sec,
    )
    logger.info("Podcast 發布完成：音訊 %s | RSS %s", audio_url, feed_path)
    return feed_path
