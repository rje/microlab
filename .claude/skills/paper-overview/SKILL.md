---
name: paper-overview
description: Read a paper's PDF and write a structured overview.json (tldr, section guide, reading focus) for the Microlab console reading workspace. Use when the user runs /paper-overview <paperId> or asks to generate or refresh a paper's AI summary.
---

# paper-overview

Generate `content/papers/<paperId>/overview.json` for a Microlab paper by reading its
PDF, so the console's reading-workspace **Summary** pane shows a per-section map of the
paper instead of just the short synopsis.

## Steps

1. **Resolve the paper.** `<paperId>` is the slug the console derives from
   `papers/manifest.json` (e.g. `mmlu`, `attention-is-all-you-need`,
   `roformer-enhanced-transformer-with-rotary-position-embedding`). Find the matching
   entry's `topic` + `filename`; the PDF is `papers/<topic>/<filename>`. If the id is
   ambiguous, grep `papers/manifest.json` for the title.
2. **Read the PDF** with the Read tool, using the `pages` parameter (max 20 pages per
   request, so read long papers in chunks). Read enough to capture the structure and
   the real findings — at least the intro, method, results, and conclusion.
3. **Write `content/papers/<paperId>/overview.json`** with exactly this shape:
   ```json
   {
     "paperId": "<paperId>",
     "generatedAt": "<YYYY-MM-DD>",
     "depthSuggestion": "implement | understand | aware",
     "tldr": "one sentence",
     "overview": "2-4 sentence plain-text overview",
     "sections": [{ "title": "<section name>", "summary": "1-2 sentences" }],
     "readingFocus": ["what to watch for", "..."]
   }
   ```
   - `sections` should follow the paper's actual section structure, in order.
   - `readingFocus` should connect to the Microlab curriculum where relevant (what this
     teaches for the eval harness, training, architecture, etc.).
   - `depthSuggestion`: `implement` for papers you'll build (architectures, training
     methods), `understand` for methods/results to grasp, `aware` for context papers.
   - Keep summaries faithful to the paper — this primes the reader, it does not replace
     reading.
4. **Validate it parses:**
   `python -c "import json; json.load(open('content/papers/<paperId>/overview.json'))"`

## Notes

- The console serves this via `GET /api/papers/<paperId>/overview`; the Summary pane
  prefers it over the short synopsis and falls back to the synopsis when it is absent.
- `overview.json` files are AI-authored content and should be committed; the user's
  `notes.md` files are git-ignored and must never be committed.
- The live service reads the file from the working tree per request, so no rebuild is
  needed for content-only changes — just make sure the file is on the deployed host.
