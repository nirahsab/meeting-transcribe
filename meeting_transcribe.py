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

Шаги 2 и 4 пропускаются при --no-diarization или MEETING_NO_DIARIZATION=1;
тогда в SRT пишется только текст без меток спикеров.

Запуск:
  python meeting_transcribe.py meeting.m4a -o meeting.srt --max-speakers 6
  python meeting_transcribe.py meeting.m4a --no-diarization
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
NO_DIARIZATION_ENV = "MEETING_NO_DIARIZATION"

# PyTorch делит каждую операцию поровну между потоками и ждёт самый медленный.
# На гибридных CPU (Intel 12+ поколения, Core Ultra) часть логических ядер —
# медленные E/LP-E-ядра и гиперпотоки, поэтому «все ядра» замедляют работу
# в разы. Ограничиваем число потоков; 8 — лучший замер на Core Ultra 7 155H.
DEFAULT_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))

# Шаг скользящего окна диаризации в долях окна (окно 10 с). Значение pyannote
# по умолчанию 0.1 (шаг 1 с) — точно, но на CPU очень медленно: на каждое окно
# считается эмбеддинг ResNet. 0.5 (шаг 5 с) — примерно в 5 раз меньше работы.
DEFAULT_DIARIZATION_STEP = 0.5


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


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def import_onnxruntime_quietly() -> None:
    """
    onnxruntime (зависимость GigaAM) при импорте ищет GPU и в WSL или на машине
    без видеокарты пишет в stderr «GPU device discovery failed». Это безвредно:
    скрипт работает на CPU. Импортируем модуль заранее с заглушенным stderr,
    дальнейшие импорты берут его из кеша и предупреждение не повторяют.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            pass
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def hf_snapshot(repo_id: str, required: tuple[str, ...] = ("config.yaml",)) -> Path:
    """
    Локальный путь к модели Hugging Face; без сети.
    Проверяет, что загрузка завершена: обязательные файлы на месте
    и в кеше нет битых ссылок на веса.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    redownload = (f"Скачайте модель заново: HF_HOME=models/huggingface hf download {repo_id} "
                  "(подробнее в README, раздел «Загрузка моделей»).")
    try:
        path = Path(snapshot_download(repo_id=repo_id, local_files_only=True))
    except LocalEntryNotFoundError:
        sys.exit(f"Модель {repo_id} не найдена в {os.environ['HF_HOME']}. {redownload}")

    # exists() следует по ссылке: битая ссылка в snapshots/ даёт False
    missing = [f for f in required if not (path / f).exists()]
    missing += [str(f.relative_to(path)) for f in path.rglob("*")
                if f.is_symlink() and not f.exists()]
    if missing:
        sys.exit(f"Модель {repo_id} загружена не полностью, нет файлов: "
                 f"{', '.join(sorted(set(missing)))}. {redownload}")
    return path


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
            min_spk: int | None, max_spk: int | None, step: float):
    import soundfile as sf
    import torch
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipeline = Pipeline.from_pretrained(pipeline_dir)
    pipeline.to(torch.device("cpu"))

    # Шаг окна задаётся только в конструкторе, поэтому меняем его у готового объекта
    seg = getattr(pipeline, "_segmentation", None)
    if seg is not None and hasattr(seg, "step"):
        pipeline.segmentation_step = step
        seg.step = step * seg.duration
        log(f"  окно {seg.duration:.0f} с, шаг {seg.step:.1f} с")

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

    with ProgressHook() as hook:
        output = pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook, **kwargs)

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
            prefix = f"[{seg.speaker}] " if seg.speaker else ""
            f.write(f"{prefix}{seg.text}\n\n")


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
    p.add_argument("--diarization-step", type=float, default=DEFAULT_DIARIZATION_STEP,
                   help="шаг окна диаризации в долях окна 10 с: 0.1 точнее, 0.5 быстрее "
                        f"(по умолчанию {DEFAULT_DIARIZATION_STEP})")
    p.add_argument("--no-diarization", action="store_true",
                   help=f"без разметки спикеров (то же, что {NO_DIARIZATION_ENV}=1)")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help=f"потоки CPU (по умолчанию {DEFAULT_THREADS}); больше — не всегда быстрее")
    args = p.parse_args()

    diarization = not (args.no_diarization or env_flag(NO_DIARIZATION_ENV))
    if not 0.05 <= args.diarization_step <= 1.0:
        sys.exit("--diarization-step должен быть в диапазоне 0.05–1.0")
    if not diarization and (args.num_speakers or args.min_speakers or args.max_speakers):
        log("Диаризация отключена: параметры числа спикеров игнорируются.")

    if not args.input.exists():
        sys.exit(f"Файл не найден: {args.input}")

    models_dir = args.models_dir.resolve()
    setup_offline(models_dir)
    gigaam_dir = check_gigaam(models_dir, args.model)
    pipeline_dir = hf_snapshot(DIARIZATION_MODEL) if diarization else None
    hf_snapshot(VAD_MODEL, required=("config.yaml", "pytorch_model.bin"))

    import_onnxruntime_quietly()
    import torch
    torch.set_num_threads(args.threads)
    log(f"Потоков CPU: {args.threads}")

    output = args.output or args.input.with_suffix(".srt")
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "audio.wav"
        log("Конвертация аудио...")
        to_wav(args.input, wav)

        turns = None
        if diarization:
            log("Диаризация (pyannote)...")
            turns = diarize(wav, pipeline_dir, args.num_speakers,
                            args.min_speakers, args.max_speakers, args.diarization_step)
            log(f"  реплик: {len(turns)}, спикеров: {len({t[2] for t in turns})}")
        else:
            log("Диаризация отключена.")

        log(f"Распознавание ({args.model})...")
        segments = transcribe(wav, args.model, gigaam_dir)
        log(f"  сегментов: {len(segments)}")

    if turns is not None:
        assign_speakers(segments, turns)
        rename_speakers(segments)
    write_srt(segments, output)
    log(f"Готово за {time.time() - t0:.0f} с -> {output}")


if __name__ == "__main__":
    main()
