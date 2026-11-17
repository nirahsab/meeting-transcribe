#!/usr/bin/env python3
"""
Локальное распознавание записи встречи.

Скрипт работает полностью офлайн и ничего не скачивает: веса моделей
заранее загружаются в папку models/ по инструкции из README.

Пайплайн:
  1. ffmpeg -> WAV 16 кГц моно
  2. pyannote (speaker-diarization-community-1) -> кто и когда говорит
  3. GigaAM-v3 e2e (transcribe_longform) -> текст с пунктуацией по сегментам
  4. Сопоставление сегментов со спикерами по максимальному перекрытию
  5. Вывод в SRT: "[Спикер N] текст"

Запуск:
  python meeting_transcribe.py meeting.m4a -o meeting.srt --max-speakers 6
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16000
DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
VAD_MODEL = "pyannote/segmentation-3.0"  # используется GigaAM в transcribe_longform
GIGAAM_MODELS = ("v3_e2e_rnnt", "v3_e2e_ctc")
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""


# ---------- утилиты ----------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def to_wav(src: Path, dst: Path) -> None:
    """Конвертация любого аудио/видео в WAV 16 кГц моно."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        check=True,
    )


def srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def setup_offline(models_dir: Path) -> None:
    """Запрет сетевых обращений. Вызывать до импорта torch/pyannote/gigaam."""
    os.environ["HF_HOME"] = str(models_dir / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYANNOTE_METRICS_ENABLED"] = "0"  # телеметрия pyannote
    os.environ.pop("HF_TOKEN", None)              # исключает загрузку внутри GigaAM


def hf_snapshot(repo_id: str) -> Path:
    """Локальный путь к модели Hugging Face; без сети."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except LocalEntryNotFoundError:
        sys.exit(f"Модель {repo_id} не найдена в {os.environ['HF_HOME']}. "
                 "Скачайте её по инструкции из README (раздел «Загрузка моделей»).")


def check_gigaam(models_dir: Path, model_name: str) -> Path:
    root = models_dir / "gigaam"
    missing = [f for f in (f"{model_name}.ckpt", f"{model_name}_tokenizer.model")
               if not (root / f).is_file()]
    if missing:
        sys.exit(f"В {root} нет файлов: {', '.join(missing)}. "
                 "Скачайте их по инструкции из README (раздел «Загрузка моделей»).")
    return root


# ---------- диаризация ----------

def diarize(wav: Path, pipeline_dir: Path, num: int | None,
            min_spk: int | None, max_spk: int | None):
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(pipeline_dir)
    pipeline.to(torch.device("cpu"))

    # Передаём waveform напрямую, чтобы не зависеть от декодера torchcodec
    data, sr = sf.read(str(wav), dtype="float32")
    waveform = torch.from_numpy(data).unsqueeze(0)

    kwargs = {}
    if num:
        kwargs["num_speakers"] = num
    else:
        if min_spk:
            kwargs["min_speakers"] = min_spk
        if max_spk:
            kwargs["max_speakers"] = max_spk

    output = pipeline({"waveform": waveform, "sample_rate": sr}, **kwargs)

    # exclusive-разметка: в каждый момент времени один спикер — удобнее для ASR
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    return [(t.start, t.end, label)
            for t, _, label in annotation.itertracks(yield_label=True)]


# ---------- распознавание ----------

def transcribe(wav: Path, model_name: str, gigaam_dir: Path) -> list[Segment]:
    import gigaam

    # Файлы уже лежат в gigaam_dir, поэтому load_model их не скачивает
    model = gigaam.load_model(model_name, device="cpu", download_root=str(gigaam_dir))
    result = model.transcribe_longform(str(wav))

    segments = []
    for seg in result:
        # Поддержка нового (объект) и старого (dict) формата ответа GigaAM
        if isinstance(seg, dict):
            text = seg["transcription"]
            start, end = seg["boundaries"]
        else:
            text, start, end = seg.text, seg.start, seg.end
        text = text.strip()
        if text:
            segments.append(Segment(float(start), float(end), text))
    return segments


# ---------- сопоставление ----------

def assign_speakers(segments: list[Segment], turns) -> None:
    for seg in segments:
        best, best_overlap = None, 0.0
        for t_start, t_end, label in turns:
            overlap = min(seg.end, t_end) - max(seg.start, t_start)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is None and turns:
            # Нет пересечения — берём ближайшую реплику по центру сегмента
            mid = (seg.start + seg.end) / 2
            best = min(turns, key=lambda t: abs((t[0] + t[1]) / 2 - mid))[2]
        seg.speaker = best or "UNKNOWN"


def rename_speakers(segments: list[Segment]) -> None:
    """SPEAKER_00 -> «Спикер 1» в порядке первого появления."""
    mapping: dict[str, str] = {}
    for seg in segments:
        if seg.speaker not in mapping:
            mapping[seg.speaker] = f"Спикер {len(mapping) + 1}"
        seg.speaker = mapping[seg.speaker]


def write_srt(segments: list[Segment], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n{srt_time(seg.start)} --> {srt_time(seg.end)}\n")
            f.write(f"[{seg.speaker}] {seg.text}\n\n")


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description="Распознавание встречи: GigaAM + pyannote -> SRT")
    p.add_argument("input", type=Path, help="аудио/видео файл встречи")
    p.add_argument("-o", "--output", type=Path, help="путь к .srt (по умолчанию рядом с входным)")
    p.add_argument("--model", default="v3_e2e_rnnt", choices=GIGAAM_MODELS,
                   help="v3_e2e_rnnt (точнее) | v3_e2e_ctc (быстрее)")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR,
                   help="папка с весами моделей (по умолчанию models/ рядом со скриптом)")
    p.add_argument("--num-speakers", type=int, help="точное число участников, если известно")
    p.add_argument("--min-speakers", type=int)
    p.add_argument("--max-speakers", type=int)
    p.add_argument("--threads", type=int, default=os.cpu_count(), help="потоки CPU")
    args = p.parse_args()

    if not args.input.exists():
        sys.exit(f"Файл не найден: {args.input}")

    models_dir = args.models_dir.resolve()
    setup_offline(models_dir)
    gigaam_dir = check_gigaam(models_dir, args.model)
    pipeline_dir = hf_snapshot(DIARIZATION_MODEL)
    hf_snapshot(VAD_MODEL)

    import torch
    torch.set_num_threads(args.threads)

    output = args.output or args.input.with_suffix(".srt")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        log("Конвертация аудио...")
        to_wav(args.input, wav)

        log("Диаризация (pyannote)...")
        turns = diarize(wav, pipeline_dir, args.num_speakers, args.min_speakers, args.max_speakers)
        log(f"  реплик: {len(turns)}, спикеров: {len({t[2] for t in turns})}")

        log(f"Распознавание ({args.model})...")
        segments = transcribe(wav, args.model, gigaam_dir)
        log(f"  сегментов: {len(segments)}")

    assign_speakers(segments, turns)
    rename_speakers(segments)
    write_srt(segments, output)
    log(f"Готово за {time.time() - t0:.0f} с -> {output}")


if __name__ == "__main__":
    main()
