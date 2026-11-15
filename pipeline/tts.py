"""階段 2：語音合成、內容雜湊快取、串接、響度標準化與 ID3 標籤。"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import subprocess
import tempfile
import time
import wave
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from mutagen.id3 import ID3, TALB, TDRC, TIT2, TPE1

from pipeline.config import AppSettings, PROJECT_ROOT, check_models_available
from pipeline.ipa import lookup_ipa
from pipeline.schema import validate_episode

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = PROJECT_ROOT / ".tts_cache"
FALLBACK_TTS_MODELS = (
    "gemini-3.8-flash-tts",
    "gemini-3.8-flash-lite-tts",
    "gemini-2.5-pro-preview-tts",
)


class TTSEngine(ABC):
    """語音合成引擎介面（可替換不同 TTS 實作）。"""

    @abstractmethod
    def synthesize(self, text: str, lang: str, voice: str, rate: float = 1.0) -> bytes:
        """將文字合成為標準 PCM WAV 位元組。"""


def _pcm_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _wav_duration_sec(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate) if rate else 0.0


def _make_silence_wav(duration_sec: float, sample_rate: int = 24000, channels: int = 1, sampwidth: int = 2) -> bytes:
    num_frames = max(0, int(round(duration_sec * sample_rate)))
    silence_pcm = b"\x00" * (num_frames * channels * sampwidth)
    return _pcm_to_wav_bytes(silence_pcm, sample_rate=sample_rate, channels=channels, sampwidth=sampwidth)


def _apply_ffmpeg_atempo(wav_bytes: bytes, rate: float, sample_rate: int = 24000) -> bytes:
    """使用 ffmpeg atempo 調整語速而不改變音高。"""
    if abs(rate - 1.0) < 1e-3:
        return wav_bytes
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = Path(tmpdir) / "in.wav"
        out_path = Path(tmpdir) / "out.wav"
        in_path.write_bytes(wav_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(in_path),
            "-filter:a",
            f"atempo={rate:.3f}",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        return out_path.read_bytes()


class GeminiTTSEngine(TTSEngine):
    """使用 Gemini TTS API 合成語音。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        sample_rate: int = 24000,
        max_chars_per_run: int = 25000,
        max_rpm: int = 9,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.sample_rate = sample_rate
        self.max_chars_per_run = max_chars_per_run
        self.max_rpm = max_rpm
        self._call_timestamps: deque[float] = deque()
        self.chars_synthesized = 0
        if client is not None:
            self.client = client
        else:
            if not api_key:
                raise RuntimeError("未設定 GEMINI_API_KEY，無法呼叫 Gemini TTS。")
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(api_version="v1beta"),
            )
            check_models_available(self.client, [self.model])

    def _pace_requests(self) -> None:
        """遵守 Tier 1 每分鐘請求上限，避免觸發 429 限流。"""
        if self.max_rpm <= 0:
            return
        now = time.monotonic()
        while self._call_timestamps and now - self._call_timestamps[0] >= 62.0:
            self._call_timestamps.popleft()
        if len(self._call_timestamps) >= self.max_rpm:
            wait_sec = 62.0 - (now - self._call_timestamps[0])
            if wait_sec > 0:
                logger.info("遵守 TTS 每分鐘請求限制，等待 %.1f 秒...", wait_sec)
                time.sleep(wait_sec)
        self._call_timestamps.append(time.monotonic())

    def synthesize(self, text: str, lang: str, voice: str, rate: float = 1.0) -> bytes:
        if self.chars_synthesized + len(text) > self.max_chars_per_run:
            raise RuntimeError(
                f"已達單次執行 TTS 字元上限（{self.max_chars_per_run} 字元），中止合成以控制費用。"
            )

        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                language_code=lang,
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                ),
            ),
        )

        last_exc: Exception | None = None
        pcm_bytes: bytes | None = None
        prompt_text = text
        fallback_iter = iter([m for m in FALLBACK_TTS_MODELS if m != self.model])
        for attempt in range(1, 6):
            try:
                self._pace_requests()
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt_text,
                    config=config,
                )
                cand = response.candidates[0] if response.candidates else None
                if cand is None or cand.content is None or not cand.content.parts:
                    reason = getattr(cand, "finish_reason", "UNKNOWN") if cand else "NO_CANDIDATES"
                    # 單字或無標點短詞有時會觸發 OTHER，加上句點有助穩定生成音訊
                    if not prompt_text.endswith((".", "。", "!", "?")):
                        prompt_text = f"{text}{'。' if lang.startswith('zh') else '.'}"
                    raise ValueError(f"TTS 未回傳音訊（finish_reason={reason}, text={text!r}）")
                pcm_bytes = cand.content.parts[0].inline_data.data
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == 5:
                    break
                err_str = str(exc)
                if "GenerateRequestsPerDay" in err_str:
                    next_model = next(fallback_iter, None)
                    if next_model:
                        logger.warning("模型 %s 已達每日請求上限，改用備援模型 %s", self.model, next_model)
                        self.model = next_model
                        self._call_timestamps.clear()
                        continue
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, flags=re.IGNORECASE)
                    delay = float(m.group(1)) + 2.0 if m else 45.0
                else:
                    delay = 2.0 * (2 ** (attempt - 1))
                logger.warning("Gemini TTS 呼叫失敗（第 %d/5 次），%.1f 秒後重試：%s", attempt, delay, exc)
                time.sleep(delay)

        if pcm_bytes is None:
            raise RuntimeError(f"Gemini TTS 連續 5 次失敗：{last_exc}") from last_exc

        self.chars_synthesized += len(text)
        wav_bytes = _pcm_to_wav_bytes(pcm_bytes, sample_rate=self.sample_rate)
        return _apply_ffmpeg_atempo(wav_bytes, rate=rate, sample_rate=self.sample_rate)


def format_spelling_prompt(word: str) -> str:
    """將單字轉為逐字母逗號分隔形式（例：'U, N, C, H, A, N, G, E, D'）。"""
    letters = [ch.upper() for ch in word if ch.isalpha()]
    return ", ".join(letters)


def compute_cache_key(engine_id: str, lang: str, voice: str, rate: float, text: str) -> str:
    """計算語音片段內容雜湊鍵值。"""
    payload = json.dumps(
        {"engine": engine_id, "lang": lang, "voice": voice, "rate": round(rate, 3), "text": text},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _synthesize_cached(
    engine: TTSEngine,
    engine_id: str,
    cache_dir: Path,
    text: str,
    lang: str,
    voice: str,
    rate: float,
) -> tuple[Path, bytes, bool]:
    """透過內容雜湊快取取得 WAV；回傳 (快取檔路徑, WAV 位元組, 是否命中快取)。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = compute_cache_key(engine_id, lang, voice, rate, text)
    cache_path = cache_dir / f"{key}.wav"
    if cache_path.exists():
        return cache_path, cache_path.read_bytes(), True
    wav_bytes = engine.synthesize(text=text, lang=lang, voice=voice, rate=rate)
    cache_path.write_bytes(wav_bytes)
    return cache_path, wav_bytes, False


def _concat_wavs_to_mp3(
    wav_chunks: list[bytes],
    out_mp3_path: Path,
    sample_rate: int = 24000,
    bitrate: str = "96k",
    target_lufs: float = -16.0,
) -> None:
    """串接所有 WAV 片段，套用響度標準化後輸出為單聲道 MP3。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        combined_wav = Path(tmpdir) / "combined.wav"
        with wave.open(str(combined_wav), "wb") as out_wf:
            out_wf.setnchannels(1)
            out_wf.setsampwidth(2)
            out_wf.setframerate(sample_rate)
            for chunk in wav_chunks:
                with wave.open(io.BytesIO(chunk), "rb") as in_wf:
                    out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))

        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(combined_wav),
            "-af",
            f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-b:a",
            bitrate,
            "-codec:a",
            "libmp3lame",
            str(out_mp3_path),
        ]
        subprocess.run(cmd, check=True)


def _write_id3_tags(mp3_path: Path, episode: dict[str, Any], show_cfg: dict[str, Any]) -> None:
    """寫入相容性最佳的 ID3v2.3 標籤至 MP3。"""
    show = show_cfg["show"]
    tags = ID3()
    tags.add(TIT2(encoding=1, text=episode["title_zh"]))
    tags.add(TPE1(encoding=1, text=show.get("author", show["name_en"])))
    tags.add(TALB(encoding=1, text=show["name_en"]))
    tags.add(TDRC(encoding=1, text=episode["date"]))
    tags.save(str(mp3_path), v2_version=3)


def build_tts_plan(episode: dict[str, Any], show_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """依節目結構與朗讀規則產生待合成與停頓的順序清單。"""
    show = show_cfg["show"]
    pauses = show_cfg["tts"]["pauses_sec"]
    plan: list[dict[str, Any]] = []

    def add_speech(seg_type: str, lang_key: str, text: str, story_idx: int | None = None, sent_id: str | None = None, extra: dict[str, Any] | None = None) -> None:
        item: dict[str, Any] = {
            "kind": "speech",
            "type": seg_type,
            "lang_key": lang_key,
            "text": text,
            "story_index": story_idx,
            "sentence_id": sent_id,
        }
        if extra:
            item.update(extra)
        plan.append(item)

    def add_pause(sec: float) -> None:
        if sec > 0:
            plan.append({"kind": "pause", "duration_sec": sec})

    # 1. 開場
    add_speech("intro", "en", f"Welcome to {episode['title_en']}.")
    add_pause(0.4)
    add_speech("intro", "zh", f"歡迎收聽 {episode['title_zh']}。")
    add_pause(float(pauses.get("after_intro", 1.0)))

    # 2. 新聞段落
    all_vocab: list[dict[str, Any]] = []
    stories = episode["stories"]
    for s_idx, story in enumerate(stories, start=1):
        add_speech("zh", "zh", f"第 {s_idx} 則新聞：{story['headline_zh']}", story_idx=s_idx)
        add_pause(0.6)

        for sent in story["sentences"]:
            sid = sent["id"]
            add_speech("en", "en", sent["en"], story_idx=s_idx, sent_id=sid)
            add_pause(float(pauses.get("after_en_sentence", 0.8)))
            add_speech("zh", "zh", sent["zh"], story_idx=s_idx, sent_id=sid)
            add_pause(float(pauses.get("after_zh_sentence", 0.6)))

            for v in sent.get("vocab", []):
                all_vocab.append(v)
                v_pause = float(pauses.get("vocab_element", 0.4))
                word = v["word"]
                spelling = format_spelling_prompt(word)

                add_speech("vocab_word", "en", word, story_idx=s_idx, sent_id=sid, extra={"word": word})
                add_pause(v_pause)
                if spelling:
                    add_speech("vocab_spell", "en", spelling, story_idx=s_idx, sent_id=sid, extra={"word": word})
                    add_pause(v_pause)
                add_speech("vocab_word_repeat", "en", word, story_idx=s_idx, sent_id=sid, extra={"word": word})
                add_pause(v_pause)
                add_speech("vocab_zh", "zh", f"這個字的中文意思是，{v['zh']}。", story_idx=s_idx, sent_id=sid, extra={"word": word})
                add_pause(v_pause)

        if s_idx < len(stories):
            add_pause(float(pauses.get("between_stories", 1.5)))

    # 3. 結尾：生難字快速複習與道別
    add_pause(float(pauses.get("before_outro", 1.0)))
    add_speech("outro", "zh", "最後讓我們快速複習今天學到的單字。")
    add_pause(0.5)
    for v in all_vocab:
        add_speech("outro_vocab_en", "en", v["word"], extra={"word": v["word"]})
        add_pause(0.4)
        add_speech("outro_vocab_zh", "zh", f"這個字的中文意思是，{v['zh']}。", extra={"word": v["word"]})
        add_pause(0.4)

    add_speech("outro", "en", f"Thanks for listening to {show['name_en']}. See you tomorrow!")
    add_pause(0.4)
    add_speech("outro", "zh", f"感謝收聽 {show['name_en']}，我們明天見。")
    return plan


def synthesize_episode(
    episode_dir: Path,
    settings: AppSettings,
    show_cfg: dict[str, Any],
    engine: TTSEngine | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """執行階段 2：讀取 episode.json、合成音訊、串接 MP3 並產出 segments.json。"""
    ep_path = episode_dir / "episode.json"
    if not ep_path.exists():
        raise FileNotFoundError(f"找不到腳本檔：{ep_path}")

    episode = json.loads(ep_path.read_text(encoding="utf-8"))
    # 若為外部供稿（例如 Gemini Spark），順手補齊空白的 CMUdict IPA 音標並清除 ** 標記
    ipa_updated = False
    for story in episode.get("stories", []):
        for sent in story.get("sentences", []):
            if isinstance(sent.get("en"), str) and "**" in sent["en"]:
                sent["en"] = sent["en"].replace("**", "").strip()
                ipa_updated = True
            for v in sent.get("vocab", []):
                if not v.get("ipa"):
                    filled = lookup_ipa(v.get("word", ""))
                    if filled:
                        v["ipa"] = filled
                        ipa_updated = True
    if ipa_updated:
        ep_path.write_text(json.dumps(episode, ensure_ascii=False, indent=2), encoding="utf-8")

    errors = validate_episode(episode, content_cfg=show_cfg.get("content"))
    if errors:
        raise ValueError("episode.json 驗證未通過，無法進入語音階段：" + "; ".join(errors))

    tts_cfg = show_cfg["tts"]
    audio_cfg = tts_cfg["audio"]
    sample_rate = int(audio_cfg.get("sample_rate", 24000))

    if engine is None:
        engine = GeminiTTSEngine(
            api_key=settings.gemini_api_key,
            model=settings.gemini_tts_model,
            sample_rate=sample_rate,
            max_chars_per_run=settings.max_tts_chars_per_run,
        )

    engine_id = f"{tts_cfg.get('engine', 'gemini')}:{settings.gemini_tts_model}"
    resolved_cache = cache_dir or DEFAULT_CACHE_DIR
    voices = tts_cfg["voices"]
    plan = build_tts_plan(episode, show_cfg)

    wav_chunks: list[bytes] = []
    segments: list[dict[str, Any]] = []
    cursor = 0.0
    cache_hits = 0
    cache_misses = 0

    for item in plan:
        if item["kind"] == "pause":
            dur = float(item["duration_sec"])
            wav_chunks.append(_make_silence_wav(dur, sample_rate=sample_rate))
            cursor += dur
            continue

        v_cfg = voices[item["lang_key"]]
        cache_path, wav_bytes, hit = _synthesize_cached(
            engine=engine,
            engine_id=engine_id,
            cache_dir=resolved_cache,
            text=item["text"],
            lang=v_cfg["lang"],
            voice=v_cfg["name"],
            rate=float(v_cfg.get("rate", 1.0)),
        )
        if hit:
            cache_hits += 1
        else:
            cache_misses += 1

        dur = _wav_duration_sec(wav_bytes)
        start_sec = round(cursor, 3)
        end_sec = round(cursor + dur, 3)
        cursor += dur
        wav_chunks.append(wav_bytes)

        seg_record: dict[str, Any] = {
            "type": item["type"],
            "lang": item["lang_key"],
            "text": item["text"],
            "start_sec": start_sec,
            "end_sec": end_sec,
            "audio_file": str(cache_path),
        }
        if item.get("story_index") is not None:
            seg_record["story_index"] = item["story_index"]
        if item.get("sentence_id") is not None:
            seg_record["sentence_id"] = item["sentence_id"]
        if item.get("word") is not None:
            seg_record["word"] = item["word"]
        segments.append(seg_record)

    mp3_path = episode_dir / "episode.mp3"
    _concat_wavs_to_mp3(
        wav_chunks=wav_chunks,
        out_mp3_path=mp3_path,
        sample_rate=sample_rate,
        bitrate=str(audio_cfg.get("mp3_bitrate", "96k")),
        target_lufs=float(audio_cfg.get("target_lufs", -16.0)),
    )
    _write_id3_tags(mp3_path, episode, show_cfg)

    meta = {
        "date": episode["date"],
        "total_duration_sec": round(cursor, 3),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "segments": segments,
    }
    seg_path = episode_dir / "segments.json"
    seg_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "階段 2 完成：產出 %s（總長 %.1f 秒，快取命中 %d / 新合成 %d）",
        mp3_path,
        cursor,
        cache_hits,
        cache_misses,
    )
    return meta
