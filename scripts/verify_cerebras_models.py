#!/usr/bin/env python3
"""Live smoke test for Cerebras models on your account.

Usage:
  CEREBRAS_API_KEY=csk_... python scripts/verify_cerebras_models.py

Exits 0 when every model in the chain responds; 1 on missing key or failures.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from provider_chain import CloudLiteLLMModel, build_cerebras_model_chain, _normalize_cerebras_model_id


def main() -> int:
    api_key = (os.getenv("CEREBRAS_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: CEREBRAS_API_KEY is not set (add to .env or export in shell)")
        return 1

    failures = 0
    for raw_model in build_cerebras_model_chain():
        model_id = _normalize_cerebras_model_id(raw_model)
        try:
            model = CloudLiteLLMModel(
                "cerebras",
                model_id=model_id,
                api_key=api_key,
                temperature=0,
                quiet=True,
            )
            reply = model.generate([{"role": "user", "content": "Reply with exactly: ok"}])
            snippet = (reply.content or "").strip().replace("\n", " ")[:120]
            print(f"PASS  {model_id}  ->  {snippet!r}")
        except Exception as error:
            failures += 1
            print(f"FAIL  {model_id}  ->  {error}")

    if failures:
        print(f"\n{failures} model(s) failed")
        return 1
    print(f"\nAll {len(build_cerebras_model_chain())} Cerebras model(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
