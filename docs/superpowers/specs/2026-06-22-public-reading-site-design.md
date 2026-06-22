# Public Reading Site (`/public`) — Design Spec

- **Date:** 2026-06-22
- **Status:** Approved, in build
- **Author:** rje + Claude

## Goal

Expose a public, unauthenticated reading site at `microlab.rje.ai/public` so a friend
can follow along: browse the curriculum's papers, read AI summaries, and read the PDFs
inline. **No notes, no progress, no private data.** The rest of the console stays fully
behind login.

## Security model (the load-bearing part)

The app's current rule is "deny everything without a session." This feature adds a
**narrow, explicit public allowlist** — only these routes skip auth:

- `GET /public` and `GET /public/<path>` (SPA shell routes) → serve the public page
- `GET /public/api/library` → JSON: papers grouped by phase + their overviews
- `GET /public/pdf/<paperId>` → serve that paper's PDF
- `GET /assets/<path>` → compiled static JS/CSS/worker (no user data; needed so the
  public page's bundle can load without a session)

Everything else — `/api/state`, `/api/papers/<id>/notes|progress|overview`,
`/api/markdown`, `/artifacts/...`, the authed SPA at `/` — keeps `@login_required`.

**Isolation by construction:** the public side uses a dedicated data function
`content.public_library(project_root)` that builds its payload **only** from
`papers/manifest.json`, `site/content/phases.json` (for phase grouping/order), and
`content/papers/<id>/overview.json`. It never reads `notes.md`, the SQLite DB, or
progress, and never calls `load_state`/`/api/state`. There is no code path from public
routes to private data because the public code does not import or touch it.

**Regression guard:** tests assert (a) each public route returns 200 with no session,
(b) private routes still 401/redirect with no session, and (c) the library payload
contains no `notes`/`progress` keys and no `notes.md` content.

## Frontend

A **separate React entry** (`site/public.html` → `site/src/public-main.tsx` →
`PublicApp.tsx`), built as a second vite input so it shares nothing with the authed
`App` bundle. It reuses the existing, verified `PdfView` (pdf.js canvas renderer) so
the public PDF experience matches the workspace.

Layout, organized **by curriculum phase** (Phase 0 → 13, then "Additional reading"):
- **Index view** (`/public`): a clean, attractive list — phase sections, each paper as
  a card (title, authors, year, TL;DR, "Read" button + arXiv link).
- **Reading view** (`/public/p/<id>`, client-routed): the paper's PDF rendered inline
  via `PdfView` on one side and the full AI summary (TL;DR, overview, section guide,
  reading focus) **alongside** it. Responsive: side-by-side on desktop, stacked/tabbed
  on mobile. Styled to read well — generous type, calm palette, matching the console.

Data: `PublicApp` fetches `GET /public/api/library` once; PDFs load from
`GET /public/pdf/<id>`. No CSRF, no credentials needed.

## Backend shape

- `src/microlab/console/content.py`: add `public_library(project_root)` and a public
  `public_pdf_path(project_root, paper_id)` helper (validates id against the manifest,
  resolves the PDF safely).
- `src/microlab/console/app.py`: add the four public route groups above, each WITHOUT
  `@login_required`; add the unauthenticated `/assets/<path>` static route (dist/assets
  only, path-safe). Leave all existing routes untouched.

## Content

Generate `overview.json` for the ~41 papers that lack one, via the `/paper-overview`
skill (read each PDF, write the structured overview), so every paper on the public site
has a real summary. Review for faithfulness before committing.

## Build order

1. Public backend + isolation tests.
2. Public React page (reuse `PdfView`), second vite entry, styling.
3. Verify auth boundary + UI with Playwright (public works sans login; private
   unreachable unauthenticated).
4. Generate all remaining overviews; review; commit.
5. Merge → deploy → verify live (`/public` open, private still gated).

## Risks

- **Static-asset exposure:** serving `/assets/*` unauthenticated exposes the compiled
  front-end bundle (no secrets; standard practice). Accepted.
- **Public PDF hosting:** re-hosts arXiv preprints on a public URL — low risk for a
  personal site; arXiv source link always shown as canonical.
- **Route precedence:** `/public...` and `/assets/...` must match before the authed
  catch-all `/<path:requested>`; Werkzeug ranks literal-prefix rules higher, and the
  isolation test confirms it.

## Self-review

- Covers the approved design: public-by-phase, PDFs via our pdf.js, summaries
  alongside, attractive, no private data. Isolation stated once and enforced by tests.
- No placeholders. Scope is one focused feature plus a content batch.
