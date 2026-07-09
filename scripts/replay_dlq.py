"""Operator CLI: inspect and replay dead-lettered A2A messages.

The DLQ preserves every failed request's full payload + history headers, so an
operator can re-drive work after fixing the cause (agent redeployed, schema
fixed). Replay re-produces the original message to its inbox topic as a fresh
attempt with a new deterministic correlation id — consumers dedup as usual.

Usage (from the repo root):

    uv run python scripts/replay_dlq.py list  [--limit 20]
    uv run python scripts/replay_dlq.py replay --cid <correlation_id>
    uv run python scripts/replay_dlq.py replay --all [--dry-run]

Requires KAFKA_ENABLED infra reachable (KAFKA_BOOTSTRAP_SERVERS etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from genie.messaging.awaiting import get_awaiting_store
from genie.messaging.broker import KafkaBroker
from genie.messaging.envelope import (
    HDR_ATTEMPT,
    HDR_CORRELATION_ID,
    HDR_ERROR,
    HDR_KIND,
    HDR_TASK_ID,
    HDR_THREAD_ID,
    HDR_TO,
    KIND_REQUEST,
    correlation_id,
    dlq_topic,
    inbox_topic,
    reply_topic,
)


async def _collect_dlq(broker: KafkaBroker, limit: int, timeout_s: float = 10.0) -> list:
    """Read up to ``limit`` records from the DLQ with a fresh (non-committing) group."""
    records = []
    gen = broker.consume([dlq_topic()], group=f"dlq-replay-{uuid.uuid4().hex}")
    try:
        while len(records) < limit:
            records.append(await asyncio.wait_for(gen.__anext__(), timeout=timeout_s))
    except (asyncio.TimeoutError, StopAsyncIteration):
        pass
    return records


async def cmd_list(args: argparse.Namespace) -> int:
    """Print the DLQ's contents: cid, target agent, attempt, error."""
    broker = KafkaBroker()
    try:
        records = await _collect_dlq(broker, args.limit)
        if not records:
            print("DLQ is empty (or unreachable).")
            return 0
        for bm in records:
            h = bm.headers
            print(f"cid={h.get(HDR_CORRELATION_ID, '?'):34s} to={h.get(HDR_TO, '?'):16s} "
                  f"attempt={h.get(HDR_ATTEMPT, '?'):>2s} task={h.get(HDR_TASK_ID, '?'):8s} "
                  f"error={h.get(HDR_ERROR, '')[:80]}")
        print(f"\n{len(records)} dead letter(s).")
        return 0
    finally:
        await broker.close()


async def cmd_replay(args: argparse.Namespace) -> int:
    """Re-produce selected dead letters to their inbox topics as fresh attempts."""
    if not args.cid and not args.all:
        print("replay needs --cid <id> or --all", file=sys.stderr)
        return 2
    broker = KafkaBroker()
    store = get_awaiting_store()
    replayed = 0
    try:
        for bm in await _collect_dlq(broker, args.limit):
            h = bm.headers
            cid = h.get(HDR_CORRELATION_ID, "")
            if args.cid and cid != args.cid:
                continue

            rec = store.get(cid)
            agent_id = h.get(HDR_TO) or (rec or {}).get("agent_id") or ""
            topic = (rec or {}).get("inbox_topic") or inbox_topic(agent_id)
            run_id = (rec or {}).get("run_id", "")
            task_id = h.get(HDR_TASK_ID) or (rec or {}).get("task_id", "")
            attempt = int(h.get(HDR_ATTEMPT, "1")) + 1
            new_cid = correlation_id(run_id, task_id, attempt) if run_id else uuid.uuid4().hex

            if args.dry_run:
                print(f"[dry-run] would replay {cid} → {topic} as attempt {attempt} (new cid {new_cid})")
                continue

            value = bm.value
            try:  # keep in-body metadata consistent with the replay attempt
                data = json.loads(value)
                data.setdefault("metadata", {})["correlation_id"] = new_cid
                value = json.dumps(data).encode("utf-8")
            except Exception:
                pass  # non-JSON poison stays as-is — the agent will re-judge it

            await broker.produce(topic, value=value, key=h.get(HDR_THREAD_ID) or None, headers={
                **h,
                HDR_KIND: KIND_REQUEST,
                HDR_CORRELATION_ID: new_cid,
                HDR_ATTEMPT: str(attempt),
                "replayed_from": cid,
                "reply_to": reply_topic(),
            })
            print(f"replayed {cid} → {topic} as attempt {attempt} (new cid {new_cid})")
            replayed += 1
            if args.cid:
                break
        print(f"\n{replayed} message(s) replayed.")
        return 0
    finally:
        await broker.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect / replay the A2A dead-letter queue")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="show dead letters")
    p_list.add_argument("--limit", type=int, default=50)
    p_list.set_defaults(fn=cmd_list)

    p_replay = sub.add_parser("replay", help="re-produce dead letters to their inbox topics")
    p_replay.add_argument("--cid", help="replay one correlation id")
    p_replay.add_argument("--all", action="store_true", help="replay everything read")
    p_replay.add_argument("--limit", type=int, default=50)
    p_replay.add_argument("--dry-run", action="store_true")
    p_replay.set_defaults(fn=cmd_replay)

    args = parser.parse_args()
    return asyncio.run(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
