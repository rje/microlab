import { useEffect, useState } from "react";

import { PdfView } from "./PdfView";

type Overview = {
  paperId: string;
  tldr?: string;
  overview?: string;
  depthSuggestion?: string;
  sections?: { title: string; summary: string }[];
  readingFocus?: string[];
};
type PaperEntry = {
  id: string;
  title: string;
  authors: string;
  year: number;
  topic: string;
  sourceUrl: string;
  pdfUrl: string;
  overview: Overview | null;
};
type PhaseGroup = { id: string; title: string; papers: PaperEntry[] };
type Library = { phases: PhaseGroup[]; additional: PaperEntry[] };

function paperIdFromPath(): string | null {
  const m = window.location.pathname.match(/^\/public\/p\/(.+)$/);
  return m ? decodeURIComponent(m[1]) : null;
}

export function PublicApp() {
  const [library, setLibrary] = useState<Library | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeId, setActiveId] = useState<string | null>(paperIdFromPath());

  useEffect(() => {
    fetch("/public/api/library")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setLibrary)
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    const onPop = () => setActiveId(paperIdFromPath());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const open = (id: string) => {
    window.history.pushState({}, "", `/public/p/${encodeURIComponent(id)}`);
    setActiveId(id);
    window.scrollTo(0, 0);
  };
  const home = () => {
    window.history.pushState({}, "", "/public");
    setActiveId(null);
  };

  if (error) {
    return <div className="pub-center">Failed to load reading list: {error}</div>;
  }
  if (!library) {
    return <div className="pub-center">Loading…</div>;
  }

  const all = [...library.phases.flatMap((p) => p.papers), ...library.additional];
  const active = activeId ? all.find((p) => p.id === activeId) ?? null : null;

  if (active) {
    return <PaperReader paper={active} onBack={home} />;
  }
  return <LibraryView library={library} onOpen={open} />;
}

function LibraryView({ library, onOpen }: { library: Library; onOpen: (id: string) => void }) {
  const groups: PhaseGroup[] = [
    ...library.phases,
    ...(library.additional.length
      ? [{ id: "additional", title: "Additional reading", papers: library.additional }]
      : [])
  ];
  return (
    <div className="pub-shell">
      <header className="pub-hero">
        <p className="pub-eyebrow">Microlab</p>
        <h1>Reading List</h1>
        <p className="pub-sub">
          Key papers on how modern LLMs are built — each with a short AI summary so you can
          follow the curriculum. Read the PDF and the summary side by side.
        </p>
      </header>
      {groups.map((g) => (
        <section className="pub-phase" key={g.id}>
          <h2>{g.title}</h2>
          <div className="pub-grid">
            {g.papers.map((p) => (
              <article className="pub-card" key={p.id}>
                <div className="pub-card-meta">
                  <span>{p.topic}</span>
                  <span>{p.year}</span>
                </div>
                <h3>{p.title}</h3>
                <p className="pub-authors">{p.authors}</p>
                {p.overview?.tldr && <p className="pub-tldr">{p.overview.tldr}</p>}
                <div className="pub-card-actions">
                  <button type="button" onClick={() => onOpen(p.id)}>
                    Read &rarr;
                  </button>
                  <a href={p.sourceUrl} target="_blank" rel="noreferrer">
                    arXiv &#8599;
                  </a>
                </div>
              </article>
            ))}
          </div>
        </section>
      ))}
      <footer className="pub-footer">Microlab · a personal LLM-from-scratch learning lab</footer>
    </div>
  );
}

function PaperReader({ paper, onBack }: { paper: PaperEntry; onBack: () => void }) {
  const [tab, setTab] = useState<"paper" | "summary">("summary");
  const ov = paper.overview;
  return (
    <div className="pub-reader">
      <header className="pub-reader-head">
        <button type="button" className="pub-back" onClick={onBack}>
          &larr; All papers
        </button>
        <div className="pub-reader-title">
          <h2>{paper.title}</h2>
          <p>
            {paper.authors} · {paper.year}
          </p>
        </div>
        <a className="pub-arxiv" href={paper.sourceUrl} target="_blank" rel="noreferrer">
          arXiv &#8599;
        </a>
      </header>
      <div className="pub-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "paper"}
          className={tab === "paper" ? "active" : ""}
          onClick={() => setTab("paper")}
        >
          Paper
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "summary"}
          className={tab === "summary" ? "active" : ""}
          onClick={() => setTab("summary")}
        >
          Summary
        </button>
      </div>
      <div className="pub-reader-body">
        <section className={`pub-pane pub-pdf-pane ${tab === "paper" ? "active" : ""}`}>
          <PdfView url={paper.pdfUrl} />
        </section>
        <section className={`pub-pane pub-summary-pane ${tab === "summary" ? "active" : ""}`}>
          {ov ? (
            <div className="pub-summary">
              {ov.tldr && <p className="pub-lede">{ov.tldr}</p>}
              {ov.overview && <p className="pub-overview">{ov.overview}</p>}
              {ov.sections && ov.sections.length > 0 && (
                <div className="pub-block">
                  <h4>Section guide</h4>
                  {ov.sections.map((s) => (
                    <div className="pub-sec" key={s.title}>
                      <strong>{s.title}</strong>
                      <p>{s.summary}</p>
                    </div>
                  ))}
                </div>
              )}
              {ov.readingFocus && ov.readingFocus.length > 0 && (
                <div className="pub-block">
                  <h4>Reading focus</h4>
                  <ul>
                    {ov.readingFocus.map((f) => (
                      <li key={f}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="pub-empty">A summary for this paper is coming soon.</p>
          )}
        </section>
      </div>
    </div>
  );
}
