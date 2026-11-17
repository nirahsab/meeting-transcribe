# meeting-transcribe

Локальное распознавание записей встреч на русском языке с разметкой участников. Работает полностью на CPU и офлайн: скрипт не обращается к сети, аудио не покидает машину.

Стек: [GigaAM-v3](https://github.com/salute-developers/GigaAM) для распознавания речи и [pyannote.audio](https://github.com/pyannote/pyannote-audio) для диаризации (определения говорящих). Результат сохраняется в формате SRT.

## Возможности

- Распознавание русской речи с пунктуацией и нормализацией текста (модели GigaAM-v3 e2e)
- Разметка спикеров: «Спикер 1», «Спикер 2» и т. д. в порядке первого появления
- Приём любого аудио- или видеоформата, который поддерживает ffmpeg (m4a, mp3, wav, mp4, webm и др.)
- Обработка длинных записей без ограничения по длительности
- Работа без GPU
- Работа без сети: веса моделей загружаются один раз при установке

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
- Для однократной загрузки моделей при установке:
  - доступ в интернет;
  - аккаунт и токен [Hugging Face](https://huggingface.co/settings/tokens) с правами read;
  - принятые условия доступа к моделям pyannote:
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

### Загрузка моделей

Скрипт не скачивает модели сам и работает только с папкой `models/` в корне проекта. Загрузите веса один раз из корня проекта при активированном venv.

Модели диаризации pyannote (нужен токен Hugging Face):

```bash
export HF_TOKEN=hf_xxx
HF_HOME=models/huggingface hf download pyannote/speaker-diarization-community-1
HF_HOME=models/huggingface hf download pyannote/segmentation-3.0
unset HF_TOKEN
```

Модель распознавания GigaAM `v3_e2e_rnnt` (по умолчанию):

> Сертификат CDN GigaAM (`cdn.chatwm.opensmodel.sberdevices.ru`), по всей видимости, выпущен российским удостоверяющим центром, которого нет в стандартных хранилищах сертификатов. Поэтому команды ниже запускаются с ключом `-k`, без проверки сертификата. Подлинность файлов модели (`.ckpt`) проверяет GigaAM: при загрузке модели он сверяет MD5 с контрольной суммой из своего исходного кода. Файлы токенизатора (`_tokenizer.model`) этой проверкой не покрыты.


```bash
mkdir -p models/gigaam
curl -kL -o models/gigaam/v3_e2e_rnnt.ckpt https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt.ckpt
curl -kL -o models/gigaam/v3_e2e_rnnt_tokenizer.model https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_rnnt_tokenizer.model
```

Модель GigaAM `v3_e2e_ctc` (только если планируете использовать быстрый режим):

```bash
curl -kL -o models/gigaam/v3_e2e_ctc.ckpt https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_ctc.ckpt
curl -kL -o models/gigaam/v3_e2e_ctc_tokenizer.model https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/v3_e2e_ctc_tokenizer.model
```

Ожидаемая структура:

```
models/
├── gigaam/
│   ├── v3_e2e_rnnt.ckpt
│   └── v3_e2e_rnnt_tokenizer.model
└── huggingface/
    └── hub/
        ├── models--pyannote--segmentation-3.0/
        └── models--pyannote--speaker-diarization-community-1/
```

Папку `models/` можно перенести на другую машину без доступа в интернет: скрипт найдёт её рядом с собой или по пути из `--models-dir`. Целостность весов GigaAM проверяется по контрольной сумме при загрузке модели.

## Использование

```bash
python meeting_transcribe.py meeting.m4a
```

Токен Hugging Face для запуска не нужен. Скрипт принудительно включает офлайн-режим (`HF_HUB_OFFLINE=1`), отключает телеметрию pyannote и завершается с подсказкой, если каких-то моделей нет в `models/`.

По умолчанию результат сохраняется рядом с входным файлом: `meeting.srt`.

### Параметры

| Параметр | Описание | По умолчанию |
|---|---|---|
| `input` | Путь к аудио- или видеофайлу | — |
| `-o`, `--output` | Путь к выходному `.srt` | рядом с входным файлом |
| `--model` | `v3_e2e_rnnt` (точнее) или `v3_e2e_ctc` (быстрее) | `v3_e2e_rnnt` |
| `--models-dir` | Папка с весами моделей | `models/` рядом со скриптом |
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
| `Модель pyannote/... не найдена` | Не выполнен шаг «Загрузка моделей» или указан неверный `--models-dir` |
| `В models/gigaam нет файлов` | Скачайте файлы `.ckpt` и `_tokenizer.model` для выбранной `--model` |
| `Model checksum failed` | Файл GigaAM скачан не полностью: удалите его и скачайте заново |
| `401` / `403` при `hf download` | Не задан `HF_TOKEN` или не приняты условия доступа к моделям pyannote на Hugging Face |
| `FileNotFoundError: ffmpeg` | ffmpeg не установлен или отсутствует в `PATH` |
| Все реплики отнесены к одному спикеру | Задайте `--num-speakers` или `--min-speakers 2` |
| Конфликт версий torch | Установите зафиксированные версии `torch==2.10.*`, `torchaudio==2.10.*` в чистом venv |

## Структура проекта

```
meeting-transcribe/
├── meeting_transcribe.py   # основной скрипт
├── models/                 # веса моделей (загружаются вручную, не хранятся в git)
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
