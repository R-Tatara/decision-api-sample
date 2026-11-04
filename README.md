![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
<img src="https://img.shields.io/badge/-Ubuntu-6F52B5.svg?logo=ubuntu&style=flat" alt="Ubuntu">

# Decision API Sample

This sample uses a RealSense camera and the OpenAI Decision API to determine whether the transparent lid on a black box is open or closed. It displays the live camera feed with classification probabilities and prints each prediction to the terminal.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- A connected RealSense camera with permission to access its color stream
- An OpenAI API key with access to the Decision API

## Installation

```bash
git clone https://github.com/R-Tatara/decision-api-sample.git
cd decision-api-sample
uv sync
```

## Usage

Set `OPENAI_API_KEY` in your environment before running the sample. For example:

```bash
export OPENAI_API_KEY="your-api-key"
uv run python3 main.py
```

Point the camera at the transparent lid on the black box. The terminal prints the predicted label (`closed`, `opened`, or `unclear`) and its confidence; the camera window shows the probabilities. Press `q` in the camera window to quit.

## License

MIT
