"""Progress logging for multi-question evaluation runs."""

from __future__ import annotations


def log_batch_start(total: int) -> None:
    print(f"\n=== Evaluating {total} questions ===")


def log_question_start(
    index: int,
    total: int,
    question: str,
    task_id: str | None = None,
) -> None:
    remaining_before = total - index + 1
    task_label = f" task={task_id[:8]}..." if task_id else ""
    print(
        f"\n[{index}/{total}] Starting{task_label} "
        f"({remaining_before} including this one)"
    )
    preview = question.strip().replace("\n", " ")
    if len(preview) > 120:
        preview = preview[:120] + "..."
    print(f"Q: {preview}")


def log_question_done(
    index: int,
    total: int,
    answer: str,
    *,
    error: str | None = None,
) -> None:
    remaining = total - index
    if error:
        print(f"[{index}/{total}] Failed: {error} ({remaining} left)")
        return
    preview = answer.strip()
    if len(preview) > 100:
        preview = preview[:100] + "..."
    print(f"[{index}/{total}] Answered: {preview!r} ({remaining} left)")


def log_batch_done(total: int, succeeded: int) -> None:
    failed = total - succeeded
    print(
        f"\n=== Done: {succeeded}/{total} answered"
        + (f", {failed} failed" if failed else "")
        + " ==="
    )
