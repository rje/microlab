---
name: paper-cards
description: Generate spaced-repetition flashcards from a paper for the Microlab console's recall system. Use when the user runs /paper-cards <paperId> or asks to generate or refresh a paper's flashcards.
---

# paper-cards

Generate `content/papers/<paperId>/cards.json` — the flashcards the console's
Recall (SM-2 spaced repetition) system reviews. Cards are git-tracked content; the
review *schedule* lives separately in SQLite, so regenerating cards is safe.

## Steps

1. **Resolve the paper.** `<paperId>` is the console slug (e.g. `mmlu`,
   `lora-low-rank-adaptation-of-large-language-models`). Its overview already
   exists at `content/papers/<paperId>/overview.json` — that distilled summary is
   the best source. Read it (and the PDF under `papers/<topic>/<filename>` if you
   need more detail).
2. **Write `content/papers/<paperId>/cards.json`** with ~8 cards. Schema:
   ```json
   {
     "paperId": "<paperId>",
     "cards": [
       {"id": "<paperId>#1", "question": "...", "answer": "..."}
     ]
   }
   ```
   - Ids are `<paperId>#1`, `<paperId>#2`, … (stable, sequential). Stable ids
     matter: the SQLite review schedule is keyed by card id, so keep existing ids
     stable when refreshing.
   - Prefer "why/how" questions that test the mechanism and findings over trivia.
     Answers are concise (1–3 sentences) and self-contained. Base everything on the
     paper/overview — never invent numbers.
3. **Validate:**
   `python -c "import json; d=json.load(open('content/papers/<paperId>/cards.json')); assert d['cards']; print(len(d['cards']))"`

## Notes

- The console serves due cards via `GET /api/recall/due` (new cards first), and
  records reviews via `POST /api/recall/review` (SM-2). No rebuild needed for
  content-only changes — the server reads the file per request.
- Cards (`cards.json`) and overviews (`overview.json`) are committed; the user's
  `notes.md` and the review DB are not.
