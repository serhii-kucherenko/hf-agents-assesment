---
title: GAIA Agent — Agents Course Unit 4
emoji: 🕵🏻‍♂️
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 5.25.2
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Agent — Hugging Face Agents Course (Unit 4)

A **smolagents**-based agent for the [Unit 4 final assignment](https://huggingface.co/learn/agents-course/unit4/introduction): answer 20 Level-1 GAIA questions via the course scoring API and reach **≥30%** for the certificate.

**Live Space:** [ken2ki/Final_Assignment_Template](https://huggingface.co/spaces/ken2ki/Final_Assignment_Template)  
**Source:** [github.com/serhii-kucherenko/Final_Assignment_Template](https://github.com/serhii-kucherenko/Final_Assignment_Template)

## What it does

1. Fetches questions (and file attachments) from the course API.
2. Runs a **CodeAgent** with web search, Wikipedia, arXiv, PDF/Excel/CSV/image/audio tools, YouTube transcripts, and local Python execution.
3. Verifies answers against GAIA formatting rules and retries once if needed.
4. Submits answers for scoring and leaderboard update.

Grading uses **exact match** against the course API — answers are normalized (no `FINAL ANSWER:` prefix in submissions).

## Architecture

```
Question → [optional strategy hints] → [optional planner] → CodeAgent + tools → Verifier → Answer → Course API
```

| Component | Role |
|-----------|------|
| `agent/gaia_agent.py` | Public entry point (`GaiaAgent`) |
| `agent/pipeline.py` | Orchestrates retriever, planner, agent, verifier, voting |
| `agent/agent_runner.py` | smolagents `CodeAgent` loop |
| `agent/tools.py` | Extended tool set (PDF, OCR, arxiv, math, etc.) |
| `agent/verifier.py` | Format checks + answer extraction |
| `app.py` | Gradio UI for Space submission |
| `run_local.py` | Local runner and score check |
| `eval/` | Fixtures and course API client for regression tests |

**Pipeline depth** (`PIPELINE_DEPTH` env var):

- `minimal` (default) — agent + verifier, one retry
- `standard` — adds lightweight planner and numeric double-check
- `full` — adds self-correction loops and optional majority vote

## Prerequisites

- Python 3.11+
- For **local** runs: one of
  - [Ollama](https://ollama.com) (e.g. `gemma4:latest`)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) + Qwen3 GGUF (see `requirements-local.txt`)
- For **HF Space**: a free [Groq API key](https://console.groq.com) (HF inference credits are easy to exhaust)

Optional system tools: `tesseract` (OCR), `ffmpeg` (audio).

## Setup (local)

```bash
git clone https://github.com/serhii-kucherenko/Final_Assignment_Template.git
cd Final_Assignment_Template
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```env
HF_USERNAME=your_hf_username

# Example: Ollama (simplest local setup)
LLM_PROVIDER=ollama
OLLAMA_MODEL=gemma4:latest
OLLAMA_API_BASE=http://127.0.0.1:11434
THINK_MODE=auto
```

## Run locally

```bash
# Quick smoke test (one trivial question)
python run_local.py --mode single

# One random course question
python run_local.py --mode random

# Full 20-question run + submit to course API for score
python run_local.py --mode score
```

Pass threshold: **6/20 correct (30%)**. The script prints `PASS` or `NOT YET PASSING` and the course API score.

### Regression tests (no LLM)

```bash
pytest tests/ -q
python eval/run_eval.py --mode fixtures --min-score 80
```

## Run on Hugging Face Space

1. Duplicate this Space or use [ken2ki/Final_Assignment_Template](https://huggingface.co/spaces/ken2ki/Final_Assignment_Template).
2. In **Settings → Secrets**, add:
   - `GROQ_API_KEY` — required for inference (free at [console.groq.com](https://console.groq.com))
   - Optional: `GROQ_MODEL=llama-3.3-70b-versatile` (default if unset)
3. Open the Space, log in with Hugging Face, click **Run Evaluation & Submit All Answers**.

> **Note:** HF Inference (`HF_TOKEN` only) often hits **402 Payment Required** when monthly credits run out. Groq bypasses that.

## Claim your certificate

After scoring **≥30%** on the course API:

1. Visit [Unit4-Final-Certificate](https://huggingface.co/spaces/agents-course/Unit4-Final-Certificate).
2. Sign in, enter your name, click **Get My Certificate**.

Local `run_local.py --mode score` counts — you do not need a successful Space run if the API already has your score under your HF username.

## Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `llamacpp` locally; Groq on Space if `GROQ_API_KEY` set | Model backend |
| `OLLAMA_MODEL` | — | Ollama model name |
| `GROQ_API_KEY` | — | Groq API key (Space) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `HF_USERNAME` | — | Your HF username for course submit |
| `PIPELINE_DEPTH` | `minimal` | Pipeline complexity |
| `THINK_MODE` | `auto` | Per-question reasoning: `auto`, `on`, `off` |
| `AGENT_MAX_STEPS` | `12` | Max CodeAgent steps per question |
| `RETRIEVER_ENABLED` | `0` | Strategy hints from past runs |
| `SELF_EVOLVE` | `0` | Write learning records after graded eval |

See `.env.example` for llama.cpp and full options.

## Data policy

This project **does not** download the GAIA benchmark dataset from Hugging Face. Evaluation uses:

- The **course scoring API** (20 questions)
- Local **fixtures** in `eval/fixtures/` for regression

Ground truth comes only from the course submit response, not from external benchmark dumps.

## Project layout

```
agent/           # Agent pipeline, tools, verifier, memory
app.py           # Gradio Space UI
run_local.py     # Local runner
eval/            # Course client, fixtures, run_eval.py
tests/           # Unit and smoke tests
tools.py         # Base smolagents tools (search, files, audio, …)
scoring.py       # Answer normalization and pass threshold
model_provider.py # llama.cpp / Ollama / Groq / HF routing
```

## License

Course assignment template — see Hugging Face Agents Course terms.
