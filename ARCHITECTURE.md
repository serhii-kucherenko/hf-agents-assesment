# Architecture

This document explains how the agent works, step by step, for someone who has never seen the codebase before.

## What this agent does

The agent answers **real-world questions** that may include **file attachments** (PDF, image, Excel, audio, etc.). Each answer should be a **short, exact value** — a number, a name, or a comma-separated list.

For each question it:

1. Reads the question text (and attachment, if any).
2. Uses tools and reasoning to find the answer.
3. Returns only the answer value (no extra explanation).
4. Can submit batches to a **scoring API**, which checks exact match against expected answers.

---

## Big picture

Think of the system as a small factory line:

```
Scoring API  →  Pipeline  →  CodeAgent + Tools  →  Verifier  →  Submit answer
     ↑              ↑                                    ↑
 questions     optional extras                    format check
 attachments    (plan, hints)                       + one retry
```

You can run this factory in two ways:

| Entry point | When you use it |
|-------------|-----------------|
| **`app.py`** | Gradio UI on Hugging Face Space — click one button to run the full batch and submit |
| **`run_local.py`** | Your laptop — same logic, good for development and faster iteration |

Both call the same core class: **`GaiaAgent`**.

---

## End-to-end flow (one question)

Here is what happens when you ask the agent a single question.

### 1. Question arrives

`GaiaAgent` receives:

- `question` — the question text
- `file_path` — optional local path to a downloaded attachment
- `task_id` — optional ID for logging and artifacts

If the question has an attachment, the file is downloaded first (`file_resolver.py` → `GET /files/{task_id}` on the scoring API).

### 2. Pipeline prepares context (`agent/pipeline.py`)

Before the language model runs, the pipeline may add extra context to the prompt:

**Strategy hints (optional)** — `agent/retriever.py`  
If `RETRIEVER_ENABLED=1`, the agent reads past runs from `memory/experience.jsonl` and injects hints like “for numeric questions, using `execute_code` worked before.” It never injects actual answers from past runs — only strategies.

**Plan (optional)** — `agent/planner.py`  
If `PIPELINE_DEPTH` is `standard` or `full`, one extra LLM call produces a short JSON plan (2–4 steps). At `minimal` depth (default), planning is skipped.

**Prompt assembly** — `agent/agent_runner.build_prompt()`  
The final prompt includes the question, attachment hints (“this is a PDF, use `read_pdf`”), and any plan/hints above.

### 3. Think mode is chosen (`agent/think_mode.py`)

Some models support extended internal reasoning. The agent turns that on or off **per question**:

- **Off** for trivial questions (“opposite of left”)
- **On** for calculations, attachments, long questions, YouTube/arXiv lookups

Controlled by `THINK_MODE=auto|on|off`.

### 4. CodeAgent runs (`agent/agent_runner.py`)

The heart of the system is a **smolagents CodeAgent**: a language model that writes **Python code blocks** to call tools, inspect results, and eventually call `final_answer("...")`.

Why code? Many questions need several steps: search the web, read a file, compute a number, then answer. The agent loops:

```
LLM writes code → tools run → results go back → LLM writes more code → … → final_answer
```

Default limit: **12 steps** (`AGENT_MAX_STEPS`).

The system prompt enforces answer format rules: short values, no “FINAL ANSWER:” prefix inside `final_answer()`, lists formatted as `a, b, c`.

### 5. Tools (`agent/tools.py` + `tools.py`)

The agent has one toolbox shared by all questions:

| Category | Tools | Typical use |
|----------|-------|-------------|
| Web | DuckDuckGo search, visit webpage, Wikipedia, fetch URL as markdown | Facts, current info |
| Academic | arXiv search | Paper titles, authors |
| Files | read PDF, Excel, CSV, text; OCR on images; transcribe audio; describe image | Attachments |
| Video | YouTube transcript | Questions about video content |
| Compute | `execute_code` (local Python/bash), math helpers (add, multiply, …) | Counting, parsing |
| Browser | Playwright (if `BROWSER_ENABLED=1`) | Heavy JavaScript pages |

Tools return **text** the model reads on the next step. Nothing runs unless the model asks for it in code.

### 6. Verifier checks the answer (`agent/verifier.py`)

The raw model output is not sent directly. The verifier:

1. **Strips** hidden reasoning blocks some models emit
2. **Extracts** the value from `final_answer("...")` or similar patterns
3. **Normalizes** via `scoring.normalize_answer()` (trim quotes, remove “FINAL ANSWER:” prefix)
4. **Checks format** — empty, multi-line, or overly long answers fail
5. **Spot-checks URLs** mentioned in the trace (optional HTTP fetch)
6. **Critic pass** (standard/full depth only) — optional second LLM asks “is this supported by evidence?”

If verification fails, the pipeline **retries once** with the verifier’s issues in the prompt. On retry, think mode is forced **on**.

### 7. Optional extras (depth flags)

| `PIPELINE_DEPTH` | What changes |
|------------------|--------------|
| **`minimal`** (default) | Agent + verifier + one retry. Fastest. |
| **`standard`** | + planner, numeric double-compute hint, optional critic |
| **`full`** | + self-correction loop (up to 3 rounds), optional majority vote (`VOTE_RUNS`) |

### 8. Answer returned and logged

The pipeline writes optional artifacts under `artifacts/{task_id}/`:

- `notes.md` — short summary
- `evidence.json` — answer, depth, think mode, etc.
- `plan.md` — if planning ran

The string returned to `app.py` or `run_local.py` is the **normalized answer only**.

### 9. Submission (batch runs)

For full evaluation, `app.py` or `run_local.py`:

1. `GET /questions` — fetch all tasks from the scoring API
2. Run the pipeline on each
3. `POST /submit` with `{username, agent_code, answers: [{task_id, submitted_answer}, …]}`
4. Scoring API returns score (% correct)

Grading is **exact string match** (case-insensitive after normalization). Ground truth stays on the server — the agent never stores expected answers locally.

---

## Language model routing (`model_provider.py`)

The same agent code runs locally and on the Space, but the **LLM backend** changes:

| Where | Default backend | How |
|-------|-----------------|-----|
| **Your machine** | llama.cpp or Ollama | Local server, no cloud credits |
| **HF Space** | Groq (if `GROQ_API_KEY` set) | Fast cloud API, free tier; throttled + auto-retry on 429 |
| **HF Space fallback** | Cerebras → Google Gemini | When Groq limits hit; add keys only — models are chosen automatically |
| **HF Space fallback** | Hugging Face Inference | Needs `HF_TOKEN`; monthly credits often run out (402 error) |

Set `LLM_PROVIDER` explicitly to override: `ollama`, `llamacpp`, `groq`, `cerebras`, `google`, or `hf`.

**Provider rotation** (`PROVIDER_FALLBACK_ENABLED=1`, default on): Cerebras models first, then Gemini, then Groq. Order via `PROVIDER_FALLBACK_ORDER=cerebras,google,groq`. Hard limits (TPD, 413, context) skip to the next slot immediately (default **5s** between calls).

---

## Project map

```
Final_Assignment_Template/
├── agent/
│   ├── gaia_agent.py      ← public API (GaiaAgent)
│   ├── pipeline.py        ← orchestrates the full flow
│   ├── agent_runner.py    ← CodeAgent + system prompt
│   ├── tools.py           ← extended tools
│   ├── verifier.py        ← answer extraction + checks
│   ├── planner.py         ← optional planning (standard/full)
│   ├── retriever.py       ← optional strategy hints
│   ├── think_mode.py      ← per-question reasoning toggle
│   ├── self_correction.py ← full-depth correction loops
│   ├── voting.py          ← full-depth majority vote
│   ├── evolve.py          ← post-eval memory (SELF_EVOLVE=1)
│   ├── code_interpreter.py← local Python/bash execution
│   ├── memory/store.py    ← experience.jsonl persistence
│   └── artifacts/writer.py← plan/notes/evidence files
├── tools.py               ← base tools (search, files, audio, …)
├── model_provider.py      ← LLM backend selection
├── provider_chain.py      ← cross-provider fallback (Groq → Cerebras → Gemini)
├── groq_model.py          ← throttling, retry, Groq model chain
├── scoring.py             ← normalize_answer
├── file_resolver.py       ← download attachments from scoring API
├── app.py                 ← Gradio Space UI
├── run_local.py           ← CLI runner + score mode
└── eval/                  ← fixtures, course client, run_eval.py
```

---

## Data policy

- **Allowed:** scoring API, local fixtures you write, your own run history for strategy hints
- **Not used:** bulk benchmark downloads, storing ground-truth answers for retrieval

---

## Configuration cheat sheet

See `.env.example` for the full list. The most important knobs:

| Variable | Effect |
|----------|--------|
| `LLM_PROVIDER` | Which LLM backend to use |
| `OLLAMA_MODEL` / `GROQ_API_KEY` | Local or Space model access |
| `GROQ_MODEL` | Groq model id (Space default: Scout 17B) |
| `GROQ_FALLBACK_ENABLED` | Auto-switch Groq models after hard limits (default 1) |
| `GROQ_MODEL_FALLBACK_CHAIN` | Comma-separated Groq fallback model ids |
| `CEREBRAS_API_KEY` / `GOOGLE_API_KEY` | Cross-provider fallback after Groq (no model config needed) |
| `PROVIDER_FALLBACK_ORDER` | Provider order, e.g. `cerebras,google,groq` |
| `GROQ_MIN_REQUEST_INTERVAL` | Seconds between cloud API calls (default 5) |
| `GROQ_MAX_RETRIES` | Retries on 429 (default 5) |
| `HF_USERNAME` | Your HF username for API submit |
| `PIPELINE_DEPTH` | `minimal` / `standard` / `full` |
| `THINK_MODE` | `auto` / `on` / `off` |
| `AGENT_MAX_STEPS` | Max tool loops per question (default 12) |
| `RETRIEVER_ENABLED` | Use past strategy hints |
| `SELF_EVOLVE` | Record learning after graded runs |

---

## Mental model for debugging

When something goes wrong, ask:

1. **Model errors (402, 404, Groq)?** → Check `model_provider.py` and Space secrets
2. **Wrong answer but agent ran?** → Read Space logs or `artifacts/` trace; verifier may have stripped a bad format
3. **Attachment ignored?** → Check file downloaded; prompt includes path and suffix hint
4. **Timeout / too slow?** → Reduce `AGENT_MAX_STEPS`, use faster model, or `PIPELINE_DEPTH=minimal`
5. **Score 0% locally but API says higher?** → Local summary uses submit response; ensure `HF_USERNAME` is set

---

## Testing without burning LLM credits

```bash
pytest tests/ -q                              # unit tests, mocked LLM
python eval/run_eval.py --mode fixtures       # hand-written Q&A pairs
python run_local.py --mode single             # one cheap smoke question
```

Fixtures live in `eval/fixtures/` — you control the expected answers.
