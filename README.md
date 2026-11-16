# meeting-transcribe

Локальное распознавание записей встреч на русском языке с разметкой участников. Работает полностью на CPU, аудио не покидает машину.

Стек: [GigaAM-v3](https://github.com/salute-developers/GigaAM) для распознавания речи и [pyannote.audio](https://github.com/pyannote/pyannote-audio) для диаризации (определения говорящих). Результат сохраняется в формате SRT.

## Возможности

- Распознавание русской речи с пунктуацией и нормализацией текста (модели GigaAM-v3 e2e)
- Разметка спикеров: «Спикер 1», «Спикер 2» и т. д. в порядке первого появления
- Приём любого аудио- или видеоформата, который поддерживает ffmpeg (m4a, mp3, wav, mp4, webm и др.)
- Обработка длинных записей без ограничения по длительности
- Работа без GPU

## Как это работает

```
входной файл
   │
   ▼
ffmpeg ──► WAV 16 кГц, моно
   │
   ├──► pyannote (speaker-diarization-community-1) ──► интервалы речи по спикерам
   │
   └──► GigaAM v3_e2e_rnnt (transcribe_longform) ──► сегменты текста с таймкодами
                                │
                                ▼
            сопоставление сегментов со спикерами по максимальному перекрытию
                                │
                                ▼
                        meeting.srt
```

## Требования

- Python 3.10 или новее
- [ffmpeg](https://ffmpeg.org/) в `PATH`
- Аккаунт и токен [Hugging Face](https://huggingface.co/settings/tokens) с правами read
- Принятые условия доступа к моделям pyannote:
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Установка

```bash
git clone https://github.com/nirahsab/meeting-transcribe.git
cd meeting-transcribe

python3 -m venv venv
source venv/bin/activate

# PyTorch для CPU
pip install torch==2.10.* torchaudio==2.10.* --index-url https://download.pytorch.org/whl/cpu

# GigaAM с зависимостями для длинных записей (включает pyannote.audio)
pip install "gigaam[longform] @ git+https://github.com/salute-developers/GigaAM.git"
```

Установка ffmpeg:

```bash
# Debian / Ubuntu
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

При первом запуске веса моделей скачиваются с Hugging Face и кешируются локально (`~/.cache`). Дальнейшая работа сети не требует.

## Использование

```bash
export HF_TOKEN=hf_xxx
python meeting_transcribe.py meeting.m4a
```

По умолчанию результат сохраняется рядом с входным файлом: `meeting.srt`.

### Параметры

| Параметр | Описание | По умолчанию |
|---|---|---|
| `input` | Путь к аудио- или видеофайлу | — |
| `-o`, `--output` | Путь к выходному `.srt` | рядом с входным файлом |
| `--model` | `v3_e2e_rnnt` (точнее) или `v3_e2e_ctc` (быстрее) | `v3_e2e_rnnt` |
| `--num-speakers` | Точное число участников | автоопределение |
| `--min-speakers` | Минимальное число участников | — |
| `--max-speakers` | Максимальное число участников | — |
| `--threads` | Число потоков CPU | все ядра |

### Примеры

```bash
# Известно точное число участников — лучший результат диаризации
python meeting_transcribe.py standup.mp4 --num-speakers 4

# Ограничение диапазона участников и явный путь вывода
python meeting_transcribe.py call.m4a -o out/call.srt --min-speakers 2 --max-speakers 6

# Быстрый режим
python meeting_transcribe.py interview.wav --model v3_e2e_ctc --threads 8
```

### Пример результата

```srt
1
00:00:01,240 --> 00:00:06,880
[Спикер 1] Коллеги, добрый день. Начинаем с итогов спринта.

2
00:00:07,310 --> 00:00:12,050
[Спикер 2] По релизу закрыли все задачи, кроме миграции базы.
```

## Производительность

На CPU основное время занимает диаризация. Скорость зависит от числа ядер и длительности записи; ориентир — время обработки сопоставимо с длительностью встречи. Для ускорения:

- задайте `--num-speakers`, если число участников известно;
- используйте `--model v3_e2e_ctc`;
- выделите больше потоков через `--threads`.

## Ограничения

- Сегмент распознавания, захватывающий смену говорящих, целиком относится к одному спикеру. При быстром диалоге возможны ошибки атрибуции на стыках реплик.
- Перекрывающаяся речь (говорят одновременно) размечается одним спикером.
- Имена участников не определяются: метки «Спикер N» нужно заменить вручную.
- Модели GigaAM-v3 обучены на русском языке; для других языков качество не гарантируется.

## Устранение неполадок

| Симптом | Причина и решение |
|---|---|
| `Не задан HF_TOKEN` | Экспортируйте токен: `export HF_TOKEN=hf_xxx` |
| `401` / `403` при загрузке pyannote | Не приняты условия доступа к моделям pyannote на Hugging Face |
| `FileNotFoundError: ffmpeg` | ffmpeg не установлен или отсутствует в `PATH` |
| Все реплики отнесены к одному спикеру | Задайте `--num-speakers` или `--min-speakers 2` |
| Конфликт версий torch | Установите зафиксированные версии `torch==2.10.*`, `torchaudio==2.10.*` в чистом venv |

## Структура проекта

```
meeting-transcribe/
├── meeting_transcribe.py   # основной скрипт
├── README.md
└── LICENSE
```

## Лицензии

- Код проекта: [MIT](LICENSE)
- GigaAM: [MIT](https://github.com/salute-developers/GigaAM/blob/main/LICENSE)
- pyannote.audio и модели диаризации распространяются на собственных условиях — ознакомьтесь с ними на страницах моделей на Hugging Face перед коммерческим использованием.

## Благодарности

- [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM)
- [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)
