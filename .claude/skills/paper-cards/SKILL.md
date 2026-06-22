---
name: paper-cards
description: Generate spaced-repetition flashcards from a paper for the Microlab console's recall system. Use when the user runs /paper-cards <paperId> or asks to generate or refresh a paper's flashcards.
---

# paper-cards

Generate `content/papers/<paperId>/cards.json` — the flashcards the console's
Recall (SM-2 spaced repetition) system reviews. Cards are git-tracked content; the
review *schedule* lives separately in SQLite.

## Source: read the paper, not just the overview

Read BOTH `content/papers/<paperId>/overview.json` (a map of the paper) AND the
relevant sections of the PDF (`papers/<topic>/<filename>`, via the Read tool's
`pages` param). The overview alone is too distilled to write accurate
*mechanics* cards — you need the paper for how things are actually computed,
scored, or constructed. Base every card strictly on the paper; never invent
numbers.

## The card mix (this is the important part)

Aim for **8–12 cards** that split across two kinds. A set that is all one kind has
poor coverage.

1. **Conceptual cards** — the paper's thesis, core findings, and why they matter.
   "Why does X fail?", "What did scaling reveal?", "Why prefer A over B?"
2. **Mechanics / implementation cards** — *how you would actually build or compute
   the thing the paper describes.* This is where most card sets are too thin. Ask
   yourself: "If I had to implement this in the Microlab eval harness / training
   loop, what would I need to know?" Examples of mechanics worth a card:
   - how a metric is computed (the pass@k estimator; how ECE bins predictions; how
     multiple-choice is scored by comparing answer-token probabilities)
   - how a dataset/task suite is constructed (task file format, the metrics used,
     train/dev/test splits, contamination safeguards like a canary string)
   - the exact algorithm/loss/update rule, prompt format, or data pipeline
   - hyperparameters that change the result and why (e.g. sampling temperature)

For an evaluation or methods paper, at least **a third of the cards should be
mechanics cards**. The goal is "I could build this," not only "I read about it."

## Card quality rules

- **One idea per card.** Never bundle two facts ("X and Y") into one card — split
  them. Compound cards recall badly under SRS.
- **No redundancy.** Do not write two cards that test the same fact from slightly
  different angles. Each card earns its place by testing something new.
- **Prefer why/how over trivia.** A bare number is a weak card; the reasoning
  behind it is a strong one. Put load-bearing numbers in the *answer*, not as the
  whole point of the card.
- **Answers are concise and self-contained** (1–3 sentences) — readable without the
  paper in front of you.

## Output

Write `content/papers/<paperId>/cards.json`:
```json
{
  "paperId": "<paperId>",
  "cards": [
    {"id": "<paperId>#1", "question": "...", "answer": "..."}
  ]
}
```
- Ids are `<paperId>#1`, `<paperId>#2`, … sequential. Ids are stable keys: the
  SQLite review schedule is keyed by card id, so when *refreshing* an existing set,
  keep an id pointing at the same idea. Changing the content under a reused id
  invalidates that card's review history — only do it on a deliberate regenerate.
- Validate:
  `python -c "import json; d=json.load(open('content/papers/<paperId>/cards.json')); assert 8 <= len(d['cards']) <= 12; print(len(d['cards']))"`

## Notes

- Served via `GET /api/recall/due` (new cards first), reviewed via
  `POST /api/recall/review` (SM-2). Content-only change — the server reads the file
  per request, no rebuild needed.
- `cards.json` and `overview.json` are committed; `notes.md` and the review DB are
  not.
