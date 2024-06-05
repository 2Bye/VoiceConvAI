# VoiceConvAI
Voice Conversation AI Task


## Task

The objective of this task is to develop a pipeline that:

Accepts an audio file recording of the conversation as input

Generates a summary of the conversation in SOAP format

## Requirements

- **Language of conversation** - English
- **Programming language** - Python

## Models

- QuartzNet - https://catalog.ngc.nvidia.com/orgs/nvidia/models/quartznet15x5
- PyAnnote - https://huggingface.co/pyannote/speaker-diarization-3.1 # optional
- omi-health/sum-small - https://huggingface.co/omi-health/sum-small
- LLama - https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

## Instalattion

I used the base image - **nvcr.io/nvidia/pytorch:24.04-py3**

```sh
apt update
apt install ffmpeg

pip install -r req.txt
```

### Pyannote

See installation guide - https://huggingface.co/pyannote/speaker-diarization-3.1

```sh
pip install pyannote.audio
```

### LLama

See installation guide https://github.com/ggerganov/llama.cpp

You need to **download** llama-2-7b-chat.Q4_K_M.gguf

```sh
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

## Results

soon

## Inference time

- CUDA: soon

- CPU: soon