"""Build a multi-turn chat SFT corpus, normalized to the {"turns": [{"user", "assistant",
optional "context"}, ...]} JSONL schema (single-turn rows are turns of length 1).

Sources — all HUMAN-written (house rule: no capability-by-distillation; GPT-generated
corpora like UltraChat/ShareGPT are forbidden):

  - OpenAssistant OASST1 (HF OpenAssistant/oasst1, train): message TREES. We extract
    linear English root->leaf conversations: at assistant branches take only the
    highest-ranked child (rank 0 is best; unranked ranks last), at prompter branches
    follow every child (different follow-ups are genuinely different conversations),
    cap depth at MAX_TURNS. Deleted/failed-review/empty/non-English messages end a path.
  - No Robots (HF HuggingFaceH4/no_robots, train): the MULTI-turn rows that
    build_sft_mix.py drops (its single-turn rows are already inside sft_mix.jsonl).
  - data/corpora/sft_mix.jsonl: the winning single-turn mix, reused as 1-turn rows.

Composition: ALL usable multi-turn (>= 2 turn) conversations + single-turn rows sized so
multi-turn is ~--multi-frac of EXAMPLES (must land in [0.3, 0.5] or the build fails
loudly). OASST paths that never got a follow-up are 1-turn conversations: still scarce
human data, so they are ALL kept on the single-turn side and only the remaining single
budget is sampled from sft_mix.

    python scripts/build_chat_mix.py --out data/corpora/chat_mix.jsonl

The mapping functions (normalize_no_robots_chat / sft_row_to_conv /
extract_oasst_conversations / compose_mix) are pure so they can be unit tested without
touching the network; the load_* wrappers are thin HF adapters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAX_TURNS = 6            # cap conversation depth (OASST trees can be deep and rambly)
MIN_MULTI_FRAC = 0.3     # the mix must be 30-50% multi-turn EXAMPLES
MAX_MULTI_FRAC = 0.5

Turn = dict[str, str]
Conv = dict[str, list[Turn]]


def normalize_no_robots_chat(row: dict) -> Conv | None:
    """No Robots `messages` -> {"turns": [...]}. Keep only clean MULTI-turn conversations:
    an optional leading SYSTEM message (every multi-turn no_robots row opens with a
    persona system prompt — it conditions the whole conversation, so it becomes the first
    turn's "context" and renders through the template's "### Input:" block), then strictly
    alternating user/assistant ending with assistant, >= 2 turns (single-turn rows already
    live in sft_mix), every message non-empty. Anything else (dangling user turns, broken
    alternation, empty content) is dropped, not repaired. Depth is capped at MAX_TURNS."""
    messages = row.get("messages") or []
    system = ""
    if messages and messages[0].get("role") == "system":
        system = (messages[0].get("content") or "").strip()
        messages = messages[1:]
    if len(messages) < 4 or len(messages) % 2 != 0:
        return None
    turns: list[Turn] = []
    for i in range(0, len(messages), 2):
        user_msg, asst_msg = messages[i], messages[i + 1]
        if user_msg.get("role") != "user" or asst_msg.get("role") != "assistant":
            return None
        user = (user_msg.get("content") or "").strip()
        assistant = (asst_msg.get("content") or "").strip()
        if not user or not assistant:
            return None
        turn: Turn = {"user": user}
        if not turns and system:
            turn["context"] = system
        turn["assistant"] = assistant
        turns.append(turn)
    return {"turns": turns[:MAX_TURNS]}


def sft_row_to_conv(row: dict) -> Conv | None:
    """An sft_mix {instruction, context, response} row as a 1-turn conversation. Context
    is preserved (the chat template renders it through the same "### Input:" block)."""
    instruction = (row.get("instruction") or "").strip()
    response = (row.get("response") or "").strip()
    if not instruction or not response:
        return None
    turn: Turn = {"user": instruction}
    context = (row.get("context") or "").strip()
    if context:
        turn["context"] = context
    turn["assistant"] = response
    return {"turns": [turn]}


def _usable(msg: dict) -> bool:
    return (msg.get("lang") == "en" and not msg.get("deleted")
            and bool((msg.get("text") or "").strip())
            and msg.get("review_result") is not False)


def _rank_key(msg: dict) -> tuple[float, str]:
    """Sort key for ranked assistant siblings: rank 0 is the community's best; unranked
    (None) sorts after every ranked sibling; message_id breaks ties deterministically."""
    rank = msg.get("rank")
    return (float("inf") if rank is None else float(rank), str(msg.get("message_id")))


def extract_oasst_conversations(messages: list[dict], max_turns: int = MAX_TURNS,
                                all_assistant_children: bool = False) -> list[Conv]:
    """Walk OASST message trees into linear conversations (see module docstring for the
    branching policy). Exact-duplicate conversations (same text under different ids) are
    emitted once."""
    children: dict[str | None, list[dict]] = {}
    for msg in messages:
        if _usable(msg):
            children.setdefault(msg.get("parent_id"), []).append(msg)

    convs: list[Conv] = []
    seen: set[str] = set()

    def emit(turns: list[Turn]) -> None:
        if not turns:
            return
        conv = {"turns": list(turns)}
        key = json.dumps(conv, sort_keys=True)
        if key not in seen:
            seen.add(key)
            convs.append(conv)

    def walk(prompter: dict, turns: list[Turn]) -> None:
        replies = [m for m in children.get(prompter["message_id"], [])
                   if m.get("role") == "assistant"]
        if not replies:
            emit(turns)
            return
        # Volume-vs-rank tradeoff: default follows only the community's best-ranked reply
        # (curated but yields ~3.7k EN paths); all_assistant_children follows EVERY reply
        # (~18k paths incl. down-ranked ones) — chosen for the 1B chat mix because measured
        # lab history (mix-SFT >> dolly-only) says volume+diversity wins at this scale.
        picks = sorted(replies, key=_rank_key) if all_assistant_children else \
            [min(replies, key=_rank_key)]
        for pick in picks:
            new_turns = turns + [{"user": prompter["text"].strip(),
                                  "assistant": pick["text"].strip()}]
            if len(new_turns) >= max_turns:
                emit(new_turns)
                continue
            follow_ups = [m for m in children.get(pick["message_id"], [])
                          if m.get("role") == "prompter"]
            if not follow_ups:
                emit(new_turns)
                continue
            for follow_up in sorted(follow_ups, key=lambda m: str(m.get("message_id"))):
                walk(follow_up, new_turns)

    for root in sorted(children.get(None, []), key=lambda m: str(m.get("message_id"))):
        if root.get("role") == "prompter":
            walk(root, [])
    return convs


def compose_mix(multi: list[Conv], single_keep: list[Conv], single_pool: list[Conv],
                multi_frac: float = 0.4, seed: int = 0) -> tuple[list[Conv], dict]:
    """ALL multi-turn (>= 2 turn) conversations + single-turn rows sized so multi-turn is
    ~multi_frac of EXAMPLES. `single_keep` (scarce human 1-turn conversations, e.g. OASST
    paths that never got a follow-up) is always fully included; the remaining single
    budget is a seeded sample from `single_pool` (the big sft_mix). Raises if the
    achievable fraction falls outside [MIN_MULTI_FRAC, MAX_MULTI_FRAC] — a silently
    skewed mix is worse than no mix."""
    import random

    n_multi = len(multi)
    if n_multi == 0:
        raise ValueError("no multi-turn conversations — nothing to build a chat mix from")
    want_single = round(n_multi * (1.0 - multi_frac) / multi_frac)
    n_filled = min(len(single_pool), max(0, want_single - len(single_keep)))
    n_single = len(single_keep) + n_filled
    frac = n_multi / (n_multi + n_single)
    if not (MIN_MULTI_FRAC <= frac <= MAX_MULTI_FRAC):
        raise ValueError(
            f"multi-turn fraction {frac:.3f} outside [{MIN_MULTI_FRAC}, {MAX_MULTI_FRAC}] "
            f"({n_multi} multi vs {len(single_keep)} kept + {len(single_pool)} pool singles)")
    rng = random.Random(seed)
    mix = multi + single_keep + rng.sample(single_pool, n_filled)
    rng.shuffle(mix)
    return mix, {"multi": n_multi, "single": n_single, "single_kept": len(single_keep),
                 "single_filled": n_filled, "total": len(mix)}


def turns_histogram(convs: list[Conv]) -> dict[int, int]:
    hist: dict[int, int] = {}
    for conv in convs:
        n = len(conv["turns"])
        hist[n] = hist.get(n, 0) + 1
    return dict(sorted(hist.items()))


def write_jsonl(rows: list[Conv], out: str | Path) -> int:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def load_oasst(split: str = "train") -> list[dict]:
    """All OASST1 message rows for a split (the tree walk needs every message up front)."""
    from datasets import load_dataset  # optional/heavy dep

    return list(load_dataset("OpenAssistant/oasst1", split=split))


def load_no_robots_chat(split: str = "train") -> list[Conv]:
    from datasets import load_dataset  # optional/heavy dep

    convs = []
    for raw in load_dataset("HuggingFaceH4/no_robots", split=split):
        conv = normalize_no_robots_chat(raw)
        if conv is not None:
            convs.append(conv)
    return convs


def load_sft_mix(path: str | Path) -> list[Conv]:
    convs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        conv = sft_row_to_conv(json.loads(line))
        if conv is not None:
            convs.append(conv)
    return convs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sft-mix", default="data/corpora/sft_mix.jsonl")
    ap.add_argument("--out", default="data/corpora/chat_mix.jsonl")
    ap.add_argument("--multi-frac", type=float, default=0.4,
                    help="target fraction of multi-turn EXAMPLES (must land in [0.3, 0.5])")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oasst-all-children", action="store_true",
                    help="follow every assistant reply (volume) instead of best-ranked only")
    args = ap.parse_args()

    oasst = extract_oasst_conversations(
        load_oasst("train"), all_assistant_children=args.oasst_all_children)
    no_robots = load_no_robots_chat("train")
    sft_singles = load_sft_mix(args.sft_mix)
    multi = [c for c in oasst + no_robots if len(c["turns"]) > 1]
    oasst_singles = [c for c in oasst if len(c["turns"]) == 1]
    mix, counts = compose_mix(multi, oasst_singles, sft_singles,
                              multi_frac=args.multi_frac, seed=args.seed)
    n = write_jsonl(mix, args.out)
    print(f"sources: oasst={len(oasst)} (multi {len(oasst) - len(oasst_singles)}, "
          f"single {len(oasst_singles)}) no_robots_multi={len(no_robots)} "
          f"sft_mix_available={len(sft_singles)}")
    print(f"composition: {counts}")
    print(f"turns histogram: {turns_histogram(mix)}")
    print(f"wrote {n} conversations -> {args.out}")


if __name__ == "__main__":
    main()
