"""管線總控進入點：支援 --date、--from-stage 與 --until。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline.config import (
    get_episode_dir,
    load_settings,
    load_show_config,
    parse_episode_date,
)
from pipeline.publish import publish_podcast
from pipeline.script import generate_episode_script
from pipeline.tts import synthesize_episode

STAGES = ("script", "tts", "align", "video", "publish")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_pipeline(
    date_arg: str = "today",
    from_stage: str = "script",
    until_stage: str = "publish",
    config_path: Path | None = None,
) -> None:
    """依序執行指定區間的管線階段。"""
    logger = logging.getLogger(__name__)
    if from_stage not in STAGES:
        raise ValueError(f"無效的 --from-stage：{from_stage}")
    if until_stage not in STAGES:
        raise ValueError(f"無效的 --until：{until_stage}")

    start_idx = STAGES.index(from_stage)
    end_idx = STAGES.index(until_stage)
    if start_idx > end_idx:
        raise ValueError(f"--from-stage ({from_stage}) 不可晚於 --until ({until_stage})")

    active_stages = STAGES[start_idx : end_idx + 1]
    episode_date = parse_episode_date(date_arg)
    episode_dir = get_episode_dir(episode_date)
    settings = load_settings()
    show_cfg = load_show_config(config_path)

    logger.info("執行日期：%s | 階段：%s -> %s", episode_date, from_stage, until_stage)

    if "script" in active_stages:
        generate_episode_script(
            episode_date=episode_date,
            episode_dir=episode_dir,
            settings=settings,
            show_cfg=show_cfg,
        )

    if "tts" in active_stages:
        tts_meta = synthesize_episode(
            episode_dir=episode_dir,
            settings=settings,
            show_cfg=show_cfg,
        )
        content_cfg = show_cfg.get("content", {})
        min_sec = float(content_cfg.get("target_duration_min_sec", 540))
        max_sec = float(content_cfg.get("target_duration_max_sec", 660))
        total_sec = float(tts_meta["total_duration_sec"])

        if "script" in active_stages and (total_sec < min_sec or total_sec > max_sec):
            hint = "每則固定 5 句" if total_sec < min_sec else "每則固定 4 句"
            logger.warning(
                "音訊總長 %.1f 秒超出目標範圍 (%.0f–%.0f 秒)，調整為「%s」重新產生一次...",
                total_sec,
                min_sec,
                max_sec,
                hint,
            )
            generate_episode_script(
                episode_date=episode_date,
                episode_dir=episode_dir,
                settings=settings,
                show_cfg=show_cfg,
                sentences_hint=hint,
            )
            tts_meta = synthesize_episode(
                episode_dir=episode_dir,
                settings=settings,
                show_cfg=show_cfg,
            )
            logger.info("調整後音訊總長：%.1f 秒", tts_meta["total_duration_sec"])

    for pending_stage in ("align", "video"):
        if pending_stage in active_stages:
            raise NotImplementedError(f"階段 '{pending_stage}' 將於第三階段實作。")

    if "publish" in active_stages:
        publish_podcast(episode_dir=episode_dir, show_cfg=show_cfg)


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="NEWS IN 10 每日英語新聞教學節目管線")
    parser.add_argument("--date", default="today", help="節目日期（today 或 YYYY-MM-DD）")
    parser.add_argument("--from-stage", default="script", choices=STAGES, help="起始階段")
    parser.add_argument("--until", default="publish", choices=STAGES, help="結束階段")
    parser.add_argument("--config", type=Path, default=None, help="自訂 show.yaml 路徑")
    args = parser.parse_args(argv)

    run_pipeline(
        date_arg=args.date,
        from_stage=args.from_stage,
        until_stage=args.until,
        config_path=args.config,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
