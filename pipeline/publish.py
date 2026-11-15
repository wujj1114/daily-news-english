"""階段 5（Podcast 部分）：上傳音訊至 GitHub Releases 並更新 GitHub Pages RSS (docs/feed.xml)。"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
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
PODCAST_NS = "https://podcastindex.org/namespace/1.0"

ET.register_namespace("itunes", ITUNES_NS)
ET.register_namespace("atom", ATOM_NS)
ET.register_namespace("podcast", PODCAST_NS)


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
    ET.SubElement(channel, f"{{{ITUNES_NS}}}summary").text = show["description"]
    ET.SubElement(
        channel,
        f"{{{ATOM_NS}}}link",
        {"href": podcast["feed_url"], "rel": "self", "type": "application/rss+xml"},
    )
    ET.SubElement(channel, f"{{{PODCAST_NS}}}locked").text = "no"
    feed_guid = str(uuid.uuid5(uuid.NAMESPACE_URL, podcast["feed_url"]))
    ET.SubElement(channel, f"{{{PODCAST_NS}}}guid").text = feed_guid
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
    ET.SubElement(new_item, "link").text = f"{podcast['site_url'].rstrip('/')}/#ep-{ep_date}"
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
    ET.SubElement(new_item, f"{{{ITUNES_NS}}}image", {"href": podcast["cover_url"]})

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


def publish_mp3_to_github_pages(
    mp3_path: Path,
    episode_date: str,
    docs_dir: Path,
    show_cfg: dict[str, Any],
) -> str:
    """將 MP3 複製至 docs/audio/YYYY-MM-DD.mp3（由 GitHub Pages 直接提供標準 audio/mpeg）。"""
    if not mp3_path.exists():
        raise FileNotFoundError(f"找不到音訊檔：{mp3_path}")

    audio_dir = docs_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    target_mp3 = audio_dir / f"{episode_date}.mp3"
    shutil.copy2(mp3_path, target_mp3)

    # 依 max_hosted_episodes 清理過舊音訊，避免超過 GitHub Pages 空間限制
    max_keep = int(show_cfg.get("podcast", {}).get("max_hosted_episodes", 60))
    mp3_files = sorted(audio_dir.glob("*.mp3"))
    if max_keep > 0 and len(mp3_files) > max_keep:
        for old_file in mp3_files[: len(mp3_files) - max_keep]:
            old_file.unlink()
            logger.info("已清理過舊音訊檔：%s", old_file.name)

    site_url = show_cfg["podcast"]["site_url"].rstrip("/")
    return f"{site_url}/audio/{episode_date}.mp3"


def write_podcast_index_html(
    docs_dir: Path,
    episode: dict[str, Any],
    show_cfg: dict[str, Any],
    audio_url: str,
) -> Path:
    """產生 GitHub Pages 節目首頁 docs/index.html。"""
    import html

    show = show_cfg["show"]
    podcast = show_cfg["podcast"]
    ep_date = html.escape(episode["date"])
    title = html.escape(show["name_zh"])
    desc = html.escape(show["description"])
    ep_title = html.escape(episode["title_zh"])
    feed_url = html.escape(podcast["feed_url"])
    cover_url = html.escape(podcast["cover_url"])
    safe_audio_url = html.escape(audio_url)

    index_html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="alternate" type="application/rss+xml" title="{title}" href="{feed_url}" />
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6; color: #1f2937; }}
    .header {{ display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap; }}
    .cover {{ width: 160px; height: 160px; border-radius: 12px; object-fit: cover; box-shadow: 0 4px 12px rgba(0,0,0,0.12); }}
    .btn {{ display: inline-block; background: #2563eb; color: #fff; padding: 0.5rem 1rem; border-radius: 8px; text-decoration: none; font-weight: 600; }}
    .card {{ margin-top: 2rem; padding: 1.25rem; border: 1px solid #e5e7eb; border-radius: 12px; }}
    audio {{ width: 100%; margin-top: 0.75rem; }}
  </style>
</head>
<body>
  <div class="header">
    <img class="cover" src="{cover_url}" alt="{title}" />
    <div>
      <h1>{title}</h1>
      <p>{desc}</p>
      <a class="btn" href="{feed_url}">訂閱 Podcast RSS</a>
    </div>
  </div>
  <div class="card" id="ep-{ep_date}">
    <h2>最新一集：{ep_title}</h2>
    <p>日期：{ep_date}</p>
    <audio controls preload="metadata" src="{safe_audio_url}"></audio>
  </div>
</body>
</html>
"""
    out_path = docs_dir / "index.html"
    out_path.write_text(index_html, encoding="utf-8")
    return out_path


def publish_podcast(
    episode_dir: Path,
    show_cfg: dict[str, Any],
    docs_dir: Path | None = None,
    assets_dir: Path | None = None,
) -> Path:
    """執行 Podcast 發布：託管 MP3、複製封面圖、更新首頁與 docs/feed.xml。"""
    ep_path = episode_dir / "episode.json"
    mp3_path = episode_dir / "episode.mp3"
    if not ep_path.exists() or not mp3_path.exists():
        raise FileNotFoundError(f"缺少 episode.json 或 episode.mp3：{episode_dir}")

    episode = json.loads(ep_path.read_text(encoding="utf-8"))
    audio_bytes = mp3_path.stat().st_size
    duration_sec = MP3(str(mp3_path)).info.length

    resolved_docs = docs_dir or (PROJECT_ROOT / "docs")
    resolved_docs.mkdir(parents=True, exist_ok=True)
    resolved_assets = assets_dir or (PROJECT_ROOT / "assets")
    cover_src = resolved_assets / "cover.jpg"
    if cover_src.exists():
        shutil.copy2(cover_src, resolved_docs / "cover.jpg")

    audio_host = show_cfg.get("podcast", {}).get("audio_host", "github_pages")
    if audio_host == "github_releases":
        audio_url = upload_mp3_to_github_release(mp3_path=mp3_path, episode_date=episode["date"])
    else:
        audio_url = publish_mp3_to_github_pages(
            mp3_path=mp3_path,
            episode_date=episode["date"],
            docs_dir=resolved_docs,
            show_cfg=show_cfg,
        )

    write_podcast_index_html(
        docs_dir=resolved_docs,
        episode=episode,
        show_cfg=show_cfg,
        audio_url=audio_url,
    )
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
