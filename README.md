---
title: GAIA Agent
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

# GAIA Agent

A **smolagents**-based question-answering agent with tools, verification, and a configurable pipeline.

**Live Space:** [ken2ki/Final_Assignment_Template](https://huggingface.co/spaces/ken2ki/Final_Assignment_Template)  
**Source:** [github.com/serhii-kucherenko/hf-agents-assesment](https://github.com/serhii-kucherenko/hf-agents-assesment)  
**How it works:** see [ARCHITECTURE.md](ARCHITECTURE.md)

## Quick start (local)

```bash
git clone https://github.com/serhii-kucherenko/hf-agents-assesment.git
cd hf-agents-assesment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set HF_USERNAME, LLM_PROVIDER, Ollama/Groq keys
python run_local.py --mode score
```

## Quick start (Hugging Face Space)

1. Add **`GROQ_API_KEY`** in [Space secrets](https://huggingface.co/spaces/ken2ki/Final_Assignment_Template/settings) (free at [console.groq.com](https://console.groq.com)).
2. Log in and click **Run Evaluation & Submit All Answers**.

## Tests

```bash
pytest tests/ -q
python eval/run_eval.py --mode fixtures --min-score 80
```

Configuration: `.env.example`
