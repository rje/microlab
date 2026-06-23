# PDF Highlights + Copy/Paste — Design Spec

- **Date:** 2026-06-23
- **Status:** Approved (built autonomously per owner instruction)

## Goal

In the **private** (authed) console, let the owner highlight passages in a paper
PDF and have those highlights persist, plus copy/paste text from the PDF. Copy/paste
also works on the public reader (harmless); highlight *persistence* is authed-only.

## Foundation: pdf.js text layer

PDF pages render to `<canvas>` (pixels — nothing selectable). Both features require a
pdf.js **text layer**: invisible, positioned text spans over each canvas. With it,
native browser selection works (copy/paste), and selecting real text yields the
geometry needed to anchor highlights.

- `PdfView` renders, per page, a `.pdf-page-wrap` (position: relative) containing the
  canvas and a `.textLayer` (pdf.js v6 `TextLayer` + the `.textLayer` CSS).
- Copy/paste is then free everywhere `PdfView` is used. The `.textLayer` CSS must live
  in **both** `styles.css` (authed bundle) and `public.css` (the isolated public
  bundle) — same lesson as the earlier scroll fix.

## Highlights (authed only)

- **Capture:** on text selection inside the PDF, a floating "Highlight" button appears.
  Click → read the selection's client rects, convert to **normalized page coordinates**
  (0–1 in the page's own space), capture the quoted text + page number, persist, draw.
- **Persist (SQLite):** `highlights(id, paper_id, page, rects_json, text, created_at)` —
  personal data. Authed + CSRF endpoints: `GET` / `POST` / `DELETE
  /api/papers/<id>/highlights`. The public side has the text layer but no highlight API.
- **Render:** each page has a `.pdf-hl-layer`; stored rects × current page size → yellow
  highlight divs. Zoom-independent (normalized), re-renders exactly on reload.
- **Panel:** a "Highlights (N)" list in the workspace — quoted text + page, click to jump
  (scroll the PDF to it), × to delete.

## Components

- Backend: `store.py` (highlights CRUD), `app.py` (3 endpoints), tests — mirrors
  notes/progress.
- Frontend: `PdfView.tsx` (text layer + per-page highlight overlay; props: `highlights`
  to render, `onSelectText` callback passed only in the authed workspace; imperative
  `scrollToHighlight`), `state.ts` helpers, `PaperWorkspace` (floating button, panel,
  jump), CSS in both bundles.

## Risks

- **pdf.js v6 text-layer alignment** is the fiddly part (the `--total-scale-factor`
  sizing, mapping selection rects ↔ normalized page coords). Mitigation: copy works even
  with rough alignment (DOM text order); highlight precision is verified **in a real
  browser with Playwright** (select → highlight → reload → jump round-trip), iterating
  on alignment until correct — not trusting unit tests for geometry.
- **Shared component:** `PdfView` is used by the authed workspace AND the public reader.
  Text layer benefits both; highlight wiring is authed-only. Don't regress the public
  reader (recently fixed for scrolling).

## Phasing

1. Text layer + copy/paste (ship first, low risk).
2. Highlights + panel (the main build, on top).
