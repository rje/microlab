import {
  Activity,
  ArrowUpRight,
  BookOpen,
  Brain,
  CheckCircle2,
  Circle,
  Clock3,
  ExternalLink,
  FileText,
  FlaskConical,
  Layers3,
  PlayCircle,
  TerminalSquare,
  X
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import {
  DueCard,
  EvalRunSummary,
  Highlight,
  MarkdownDocument,
  MicrolabState,
  Paper,
  PaperOverview,
  PaperSynopsis,
  Phase,
  PhaseTask,
  addHighlight,
  deleteHighlight,
  fetchDueCards,
  fetchHighlights,
  fetchMarkdownDocument,
  fetchMicrolabState,
  fetchNotes,
  fetchOverview,
  saveNotes,
  saveProgress,
  saveTaskStatus,
  submitReview
} from "./state";
import { MarkdownDocumentView } from "./MarkdownDocumentView";
import { PdfView, type PdfSelection, type PdfViewHandle } from "./PdfView";

const READ_STATES = ["unread", "skimming", "mapped", "built", "mastered"];
const DEPTHS = ["implement", "understand", "aware"];

type AppProps = {
  initialState?: MicrolabState;
};

const phaseStatusLabel: Record<Phase["status"], string> = {
  current: "Current",
  planned: "Planned",
  complete: "Complete"
};

const taskStatusLabel: Record<PhaseTask["status"], string> = {
  done: "Done",
  active: "Active",
  queued: "Queued",
  blocked: "Blocked"
};

function statusIcon(status: PhaseTask["status"]) {
  if (status === "done") {
    return <CheckCircle2 aria-hidden="true" />;
  }
  if (status === "active") {
    return <PlayCircle aria-hidden="true" />;
  }
  if (status === "blocked") {
    return <Circle aria-hidden="true" />;
  }
  return <Clock3 aria-hidden="true" />;
}

function taskProgress(tasks: PhaseTask[]) {
  if (tasks.length === 0) {
    return 0;
  }
  const completed = tasks.filter((task) => task.status === "done").length;
  return Math.round((completed / tasks.length) * 100);
}

function metricLabel(value: number) {
  if (value >= 0 && value <= 1) {
    return `${Math.round(value * 100)}%`;
  }
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value);
}

function basename(path: string) {
  const parts = path.split("/");
  return parts[parts.length - 1] || path;
}

function markdownPathFromHref(href: string) {
  const withoutQuery = href.split("?")[0].split("#")[0];
  const cleanPath = withoutQuery.replace(/^\/+/, "");
  if (cleanPath.startsWith("artifacts/")) {
    return cleanPath.replace(/^artifacts\//, "");
  }
  return cleanPath;
}

function isMarkdownHref(href: string) {
  return markdownPathFromHref(href).toLowerCase().endsWith(".md");
}

export function App({ initialState }: AppProps) {
  const [state, setState] = useState<MicrolabState | null>(initialState ?? null);
  const [error, setError] = useState<string | null>(null);
  const [activePhaseId, setActivePhaseId] = useState(initialState?.phases[0]?.id ?? "phase-0");
  const [markdownDocument, setMarkdownDocument] = useState<MarkdownDocument | null>(null);
  const [markdownError, setMarkdownError] = useState<string | null>(null);
  const [markdownLoadingPath, setMarkdownLoadingPath] = useState<string | null>(null);
  const [activePaperId, setActivePaperId] = useState<string | null>(null);
  const [recallOpen, setRecallOpen] = useState(false);

  useEffect(() => {
    if (initialState) {
      return;
    }

    fetchMicrolabState()
      .then((nextState) => {
        setState(nextState);
        setActivePhaseId(nextState.phases[0]?.id ?? "phase-0");
      })
      .catch((nextError: Error) => {
        setError(nextError.message);
      });
  }, [initialState]);

  if (error) {
    return (
      <main className="center-state">
        <FlaskConical aria-hidden="true" />
        <h1>Microlab Console</h1>
        <p>{error}</p>
      </main>
    );
  }

  if (!state) {
    return (
      <main className="center-state">
        <Activity aria-hidden="true" />
        <h1>Microlab Console</h1>
        <p>Loading project state</p>
      </main>
    );
  }

  const activePhase = state.phases.find((phase) => phase.id === activePhaseId) ?? state.phases[0];
  const papersById = new Map(state.papers.map((paper) => [paper.id, paper]));
  const readingPapers = activePhase.readingPaperIds
    .map((paperId) => papersById.get(paperId))
    .filter((paper): paper is Paper => Boolean(paper));
  const phaseRuns = state.evalRuns.filter((run) => run.phaseId === activePhase.id);
  const openMarkdownDocument = async (href: string) => {
    const path = markdownPathFromHref(href);
    setMarkdownDocument(null);
    setMarkdownError(null);
    setMarkdownLoadingPath(path);
    try {
      setMarkdownDocument(await fetchMarkdownDocument(path));
    } catch (nextError) {
      setMarkdownError(nextError instanceof Error ? nextError.message : "Unknown markdown error");
    } finally {
      setMarkdownLoadingPath(null);
    }
  };
  const closeMarkdownDocument = () => {
    setMarkdownDocument(null);
    setMarkdownError(null);
    setMarkdownLoadingPath(null);
  };

  const updatePaperProgress = (
    paperId: string,
    progress: { readState: string; depth: string | null }
  ) => {
    setState((prev) =>
      prev
        ? { ...prev, papers: prev.papers.map((p) => (p.id === paperId ? { ...p, progress } : p)) }
        : prev
    );
  };
  const updateTaskStatus = (phaseId: string, taskId: string, status: string) => {
    setState((prev) =>
      prev
        ? {
            ...prev,
            phases: prev.phases.map((ph) =>
              ph.id === phaseId
                ? { ...ph, tasks: ph.tasks.map((t) => (t.id === taskId ? { ...t, status: status as PhaseTask["status"] } : t)) }
                : ph
            )
          }
        : prev
    );
  };
  const activePaper = activePaperId
    ? state.papers.find((p) => p.id === activePaperId) ?? null
    : null;

  return (
    <>
      <div className="app-shell">
        <PhaseRail
          activePhaseId={activePhase.id}
          phases={state.phases}
          onSelectPhase={setActivePhaseId}
        />
        <main className="workspace">
          <PhaseHeader phase={activePhase} />
          <ProgressBand phase={activePhase} papers={readingPapers} runs={phaseRuns} />
          <TaskBoard
            tasks={activePhase.tasks}
            phaseId={activePhase.id}
            csrfToken={state.csrfToken ?? ""}
            onOpenMarkdown={openMarkdownDocument}
            onChangeStatus={updateTaskStatus}
          />
        </main>
        <aside className="right-rail">
          <RecallPanel onStart={() => setRecallOpen(true)} />
          <ReadingPanel
            papers={readingPapers}
            synopses={state.synopses}
            onOpen={setActivePaperId}
          />
          <ResultsPanel runs={phaseRuns} onOpenMarkdown={openMarkdownDocument} />
        </aside>
      </div>
      <MarkdownDocumentView
        document={markdownDocument}
        error={markdownError}
        loadingPath={markdownLoadingPath}
        onClose={closeMarkdownDocument}
      />
      {activePaper && (
        <PaperWorkspace
          paper={activePaper}
          synopsis={state.synopses[activePaper.id]}
          csrfToken={state.csrfToken ?? ""}
          onClose={() => setActivePaperId(null)}
          onProgressChange={updatePaperProgress}
        />
      )}
      {recallOpen && (
        <RecallSession csrfToken={state.csrfToken ?? ""} onClose={() => setRecallOpen(false)} />
      )}
    </>
  );
}

function PhaseRail({
  activePhaseId,
  phases,
  onSelectPhase
}: {
  activePhaseId: string;
  phases: Phase[];
  onSelectPhase: (phaseId: string) => void;
}) {
  return (
    <nav className="phase-rail" aria-label="Microlab phases">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <TerminalSquare />
        </div>
        <div>
          <p className="eyebrow">Microlab</p>
          <h2>Console</h2>
        </div>
      </div>

      <div className="phase-list">
        {phases.map((phase, index) => (
          <button
            className={`phase-nav ${phase.id === activePhaseId ? "is-active" : ""}`}
            key={phase.id}
            onClick={() => onSelectPhase(phase.id)}
            type="button"
          >
            <span className={`phase-number phase-${phase.status}`}>{String(index).padStart(2, "0")}</span>
            <span>
              <strong>{phase.title.replace(/^Phase \d+: /, "")}</strong>
              <small>{phaseStatusLabel[phase.status]}</small>
            </span>
          </button>
        ))}
      </div>
    </nav>
  );
}

function PhaseHeader({ phase }: { phase: Phase }) {
  return (
    <header className="phase-header">
      <div>
        <p className="eyebrow">{phaseStatusLabel[phase.status]} phase</p>
        <h1>{phase.title}</h1>
        <p className="goal">{phase.goal}</p>
      </div>
      <div className="phase-badge">
        <Layers3 aria-hidden="true" />
        <span>{phase.tasks.length || "0"} work items</span>
      </div>
    </header>
  );
}

function ProgressBand({
  phase,
  papers,
  runs
}: {
  phase: Phase;
  papers: Paper[];
  runs: EvalRunSummary[];
}) {
  const progress = taskProgress(phase.tasks);
  return (
    <section className="progress-band" aria-label="Phase status">
      <div className="progress-copy">
        <p>{phase.summary || phase.goal}</p>
      </div>
      <div className="progress-metrics">
        <MetricTile label="Work complete" value={`${progress}%`} tone="teal" />
        <MetricTile label="Reading queue" value={String(papers.length)} tone="amber" />
        <MetricTile label="Eval runs" value={String(runs.length)} tone="blue" />
      </div>
      <div className="timeline" aria-label={`${progress}% of work items complete`}>
        {phase.tasks.map((task) => (
          <span className={`timeline-step is-${task.status}`} key={task.id} title={task.title} />
        ))}
      </div>
    </section>
  );
}

function MetricTile({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={`metric-tile tone-${tone}`}>
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function TaskBoard({
  tasks,
  phaseId,
  csrfToken,
  onOpenMarkdown,
  onChangeStatus
}: {
  tasks: PhaseTask[];
  phaseId: string;
  csrfToken: string;
  onOpenMarkdown: (href: string) => void;
  onChangeStatus: (phaseId: string, taskId: string, status: string) => void;
}) {
  const groupedTasks = useMemo(() => {
    const order: PhaseTask["status"][] = ["active", "queued", "done", "blocked"];
    return order
      .map((status) => ({ status, tasks: tasks.filter((task) => task.status === status) }))
      .filter((group) => group.tasks.length > 0);
  }, [tasks]);

  return (
    <section className="task-board" aria-labelledby="task-board-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Phase work</p>
          <h2 id="task-board-heading">Execution Board</h2>
        </div>
        <span>{tasks.length} items</span>
      </div>

      <div className="task-columns">
        {groupedTasks.map((group) => (
          <div className="task-column" key={group.status}>
            <h3>{taskStatusLabel[group.status]}</h3>
            {group.tasks.map((task) => (
              <article className={`task-card is-${task.status}`} key={task.id}>
                <div className="task-status">{statusIcon(task.status)}</div>
                <div>
                  <h4>{task.title}</h4>
                  <p>{task.why}</p>
                  <label className="task-status-select">
                    <span className="visually-hidden">Status for {task.title}</span>
                    <select
                      aria-label={`Status for ${task.title}`}
                      value={task.status}
                      onChange={(event) => {
                        const next = event.target.value;
                        onChangeStatus(phaseId, task.id, next);
                        saveTaskStatus(phaseId, task.id, csrfToken, next).catch(() => {});
                      }}
                    >
                      {(["queued", "active", "done", "blocked"] as const).map((s) => (
                        <option key={s} value={s}>{taskStatusLabel[s]}</option>
                      ))}
                    </select>
                  </label>
                  {task.links.length > 0 && (
                    <div className="inline-links">
                      {task.links.map((link) => (
                        <a
                          href={`/${link}`}
                          key={link}
                          onClick={(event) => {
                            if (!isMarkdownHref(link)) {
                              return;
                            }
                            event.preventDefault();
                            onOpenMarkdown(link);
                          }}
                        >
                          <FileText aria-hidden="true" />
                          <span>{basename(link)}</span>
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}
          </div>
        ))}
      </div>
    </section>
  );
}

function ReadingPanel({
  papers,
  synopses,
  onOpen
}: {
  papers: Paper[];
  synopses: MicrolabState["synopses"];
  onOpen: (paperId: string) => void;
}) {
  return (
    <section className="rail-section" aria-labelledby="reading-heading">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Reading</p>
          <h2 id="reading-heading">Paper Queue</h2>
        </div>
        <BookOpen aria-hidden="true" />
      </div>

      <div className="paper-list">
        {papers.map((paper) => (
          <PaperCard key={paper.id} paper={paper} synopsis={synopses[paper.id]} onOpen={onOpen} />
        ))}
      </div>
    </section>
  );
}

function PaperCard({
  paper,
  synopsis,
  onOpen
}: {
  paper: Paper;
  synopsis: PaperSynopsis | undefined;
  onOpen: (paperId: string) => void;
}) {
  const readState = paper.progress?.readState ?? "unread";
  const depth = paper.progress?.depth ?? null;
  return (
    <article className="paper-card">
      <div className="paper-meta">
        <span>{paper.topic}</span>
        <span>{paper.year}</span>
      </div>
      <h3>{paper.title}</h3>
      <p className="authors">{paper.authors}</p>
      <div className="card-status">
        <span className={`rs-badge rs-${readState}`}>{readState}</span>
        {depth && <span className="depth-badge">{depth}</span>}
      </div>
      {synopsis && <p className="synopsis-lede">{synopsis.oneSentence}</p>}
      <div className="paper-actions">
        <button type="button" className="open-workspace" onClick={() => onOpen(paper.id)}>
          <BookOpen aria-hidden="true" />
          Read &amp; take notes
        </button>
        <a href={paper.sourceUrl} rel="noreferrer" target="_blank">
          <ExternalLink aria-hidden="true" />
          Source
        </a>
      </div>
    </article>
  );
}

function PaperWorkspace({
  paper,
  synopsis,
  csrfToken,
  onClose,
  onProgressChange
}: {
  paper: Paper;
  synopsis: PaperSynopsis | undefined;
  csrfToken: string;
  onClose: () => void;
  onProgressChange: (paperId: string, progress: { readState: string; depth: string | null }) => void;
}) {
  const [readState, setReadState] = useState(paper.progress?.readState ?? "unread");
  const [depth, setDepth] = useState<string>(paper.progress?.depth ?? "");
  const [notes, setNotes] = useState("");
  const [notesLoaded, setNotesLoaded] = useState(false);
  const [saved, setSaved] = useState("");
  const [tab, setTab] = useState<"paper" | "summary" | "notes">("paper");
  const [overview, setOverview] = useState<PaperOverview | null>(null);

  useEffect(() => {
    let active = true;
    fetchNotes(paper.id)
      .then((content) => {
        if (active) {
          setNotes(content);
          setNotesLoaded(true);
        }
      })
      .catch(() => {
        if (active) {
          setNotesLoaded(true);
        }
      });
    return () => {
      active = false;
    };
  }, [paper.id]);

  useEffect(() => {
    let active = true;
    fetchOverview(paper.id).then((data) => {
      if (active) {
        setOverview(data);
      }
    });
    return () => {
      active = false;
    };
  }, [paper.id]);

  useEffect(() => {
    if (!notesLoaded) {
      return;
    }
    const handle = setTimeout(() => {
      saveNotes(paper.id, csrfToken, notes)
        .then(() => setSaved("saved"))
        .catch((error: Error) => setSaved(error.message));
    }, 800);
    return () => clearTimeout(handle);
  }, [notes, notesLoaded, csrfToken, paper.id]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const persistProgress = (next: { readState?: string; depth?: string }) => {
    const rs = next.readState ?? readState;
    const d = next.depth ?? depth;
    const progress = { readState: rs, depth: d === "" ? null : d };
    saveProgress(paper.id, csrfToken, progress)
      .then(() => setSaved("saved"))
      .catch((error: Error) => setSaved(error.message));
    onProgressChange(paper.id, progress);
  };

  const pdfRef = useRef<PdfViewHandle>(null);
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const [selection, setSelection] = useState<PdfSelection | null>(null);

  useEffect(() => {
    let active = true;
    fetchHighlights(paper.id)
      .then((hls) => active && setHighlights(hls))
      .catch(() => active && setHighlights([]));
    return () => {
      active = false;
    };
  }, [paper.id]);

  const commitHighlight = () => {
    if (!selection) {
      return;
    }
    const sel = selection;
    setSelection(null);
    window.getSelection()?.removeAllRanges();
    addHighlight(paper.id, csrfToken, { page: sel.page, rects: sel.rects, text: sel.text })
      .then((hl) => setHighlights((prev) => [...prev, hl]))
      .catch((error: Error) => setSaved(error.message));
  };

  const removeHighlight = (id: number) => {
    setHighlights((prev) => prev.filter((h) => h.id !== id));
    deleteHighlight(paper.id, csrfToken, id).catch(() => {});
  };

  const jumpToHighlight = (hl: Highlight) => {
    setTab("paper");
    requestAnimationFrame(() => pdfRef.current?.scrollToHighlight(hl.page, hl.rects[0]?.y ?? 0));
  };

  return (
    <div className="ws-overlay" role="dialog" aria-modal="true" aria-label={`Reading ${paper.title}`}>
      {selection && (
        <button
          type="button"
          className="hl-fab"
          style={{ left: selection.anchor.x, top: selection.anchor.y }}
          onMouseDown={(event) => event.preventDefault()}
          onClick={commitHighlight}
        >
          Highlight
        </button>
      )}
      <header className="ws-header">
        <div className="ws-title">
          <h2>{paper.title}</h2>
          <p className="authors">
            {paper.authors} · {paper.year}
          </p>
        </div>
        <div className="ws-controls">
          <select
            aria-label="Reading state"
            value={readState}
            onChange={(event) => {
              setReadState(event.target.value);
              persistProgress({ readState: event.target.value });
            }}
          >
            {READ_STATES.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          <select
            aria-label="Depth"
            value={depth}
            onChange={(event) => {
              setDepth(event.target.value);
              persistProgress({ depth: event.target.value });
            }}
          >
            <option value="">depth…</option>
            {DEPTHS.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
          {saved && <span className="save-state">{saved}</span>}
          <button type="button" className="ws-close" onClick={onClose} aria-label="Close reader">
            <X aria-hidden="true" />
          </button>
        </div>
      </header>

      <div className="ws-tabs" role="tablist">
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
        <button
          type="button"
          role="tab"
          aria-selected={tab === "notes"}
          className={tab === "notes" ? "active" : ""}
          onClick={() => setTab("notes")}
        >
          Notes
        </button>
      </div>

      <div className="ws-body">
        <section className={`ws-pane ws-paper ${tab === "paper" ? "active" : ""}`}>
          <div className="ws-paper-bar">
            <a href={paper.pdfUrl} target="_blank" rel="noreferrer">
              <ExternalLink aria-hidden="true" />
              Open PDF
            </a>
          </div>
          <PdfView
            ref={pdfRef}
            url={paper.pdfUrl}
            highlights={highlights}
            onSelect={setSelection}
          />
        </section>
        <section
          className={`ws-pane ws-side ${tab === "summary" || tab === "notes" ? "active" : ""}`}
        >
          <details className="ws-highlights">
            <summary>Highlights ({highlights.length})</summary>
            {highlights.length === 0 ? (
              <p className="ws-hl-empty">Select text in the PDF, then click “Highlight”.</p>
            ) : (
              highlights.map((hl) => (
                <div className="ws-hl-item" key={hl.id}>
                  <button type="button" className="jump" onClick={() => jumpToHighlight(hl)}>
                    {hl.text.length > 140 ? `${hl.text.slice(0, 140)}…` : hl.text}
                  </button>
                  <span className="hl-page">p.{hl.page}</span>
                  <button
                    type="button"
                    className="ws-hl-del"
                    aria-label="Delete highlight"
                    onClick={() => removeHighlight(hl.id)}
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </details>
          <div className={`ws-summary ${tab === "summary" ? "active" : ""}`}>
            {overview ? (
              <>
                <p className="synopsis-lede">{overview.tldr}</p>
                {overview.overview && <p>{overview.overview}</p>}
                {overview.sections.length > 0 && (
                  <div className="study-block">
                    <strong>Section guide</strong>
                    <div className="overview-sections">
                      {overview.sections.map((section) => (
                        <div className="overview-section" key={section.title}>
                          <h4>{section.title}</h4>
                          <p>{section.summary}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {overview.readingFocus.length > 0 && (
                  <div className="study-block">
                    <strong>Reading focus</strong>
                    <ul className="study-list">
                      {overview.readingFocus.map((focus) => (
                        <li key={focus}>{focus}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : synopsis ? (
              <>
                <p className="synopsis-lede">{synopsis.oneSentence}</p>
                <div className="study-block">
                  <strong>Summary</strong>
                  <p>{synopsis.summary}</p>
                </div>
                <div className="study-block">
                  <strong>Core ideas</strong>
                  <ul className="study-list">
                    {synopsis.coreIdeas.map((idea) => (
                      <li key={idea}>{idea}</li>
                    ))}
                  </ul>
                </div>
                <div className="study-block">
                  <strong>Why it matters</strong>
                  <p>{synopsis.whyItMatters}</p>
                </div>
                <div className="study-block">
                  <strong>Reading focus</strong>
                  <ul className="study-list">
                    {synopsis.suggestedReadingFocus.map((focus) => (
                      <li key={focus}>{focus}</li>
                    ))}
                  </ul>
                </div>
              </>
            ) : (
              <p className="ws-empty">
                No AI summary yet for this paper. Generate one with{" "}
                <code>/paper-overview {paper.id}</code>.
              </p>
            )}
          </div>
          <div className={`ws-notes ${tab === "notes" ? "active" : ""}`}>
            <strong>My notes</strong>
            {notesLoaded ? (
              <textarea
                className="ws-notes-editor"
                aria-label="Notes"
                value={notes}
                placeholder="Mechanism in your own words, what not to copy, questions…"
                onChange={(event) => setNotes(event.target.value)}
              />
            ) : (
              <p className="notes-loading">Loading notes…</p>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function RecallPanel({ onStart }: { onStart: () => void }) {
  const [total, setTotal] = useState(0);
  useEffect(() => {
    let active = true;
    fetchDueCards()
      .then((d) => active && setTotal(d.total ?? 0))
      .catch(() => active && setTotal(0));
    return () => {
      active = false;
    };
  }, []);
  return (
    <section className="rail-section" aria-labelledby="recall-heading">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Recall</p>
          <h2 id="recall-heading">Flashcards</h2>
        </div>
        <Brain aria-hidden="true" />
      </div>
      <div className="recall-due">
        <strong>{total}</strong>
        <span>cards due</span>
      </div>
      <button type="button" className="recall-start" disabled={total === 0} onClick={onStart}>
        {total === 0 ? "All caught up" : "Start review"}
      </button>
    </section>
  );
}

function RecallSession({ csrfToken, onClose }: { csrfToken: string; onClose: () => void }) {
  const [cards, setCards] = useState<DueCard[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetchDueCards()
      .then((d) => active && setCards(d.cards))
      .catch((e: Error) => active && setError(e.message));
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const card = cards && idx < cards.length ? cards[idx] : null;
  const grade = (g: number) => {
    if (!card) {
      return;
    }
    submitReview(card.id, card.paperId, csrfToken, g).catch(() => {});
    setRevealed(false);
    setIdx((i) => i + 1);
  };

  return (
    <div className="recall-overlay" role="dialog" aria-modal="true" aria-label="Flashcard review">
      <div className="recall-modal">
        <header className="recall-modal-head">
          <span>{cards ? (card ? `${idx + 1} / ${cards.length}` : "Done") : "Loading"}</span>
          <button type="button" className="recall-close" onClick={onClose} aria-label="Close review">
            <X aria-hidden="true" />
          </button>
        </header>
        {error && <p className="recall-error">{error}</p>}
        {cards && !card && (
          <div className="recall-done">
            <strong>All reviewed.</strong>
            <p>Come back tomorrow for the next batch.</p>
            <button type="button" onClick={onClose}>Close</button>
          </div>
        )}
        {card && (
          <>
            <p className="recall-paper">{card.paperTitle}</p>
            <div className="recall-question">{card.question}</div>
            {revealed ? (
              <>
                <div className="recall-answer">{card.answer}</div>
                <div className="recall-grades">
                  <button type="button" onClick={() => grade(1)}>Again</button>
                  <button type="button" onClick={() => grade(3)}>Hard</button>
                  <button type="button" onClick={() => grade(4)}>Good</button>
                  <button type="button" onClick={() => grade(5)}>Easy</button>
                </div>
              </>
            ) : (
              <button type="button" className="recall-reveal" onClick={() => setRevealed(true)}>
                Show answer
              </button>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function ResultsPanel({
  runs,
  onOpenMarkdown
}: {
  runs: EvalRunSummary[];
  onOpenMarkdown: (href: string) => void;
}) {
  return (
    <section className="rail-section results-section" aria-labelledby="results-heading">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Evidence</p>
          <h2 id="results-heading">Eval Runs</h2>
        </div>
        <Activity aria-hidden="true" />
      </div>

      {runs.length === 0 ? (
        <div className="empty-results">
          <strong>Awaiting first eval run</strong>
          <p>Phase results will appear here from saved summary artifacts.</p>
        </div>
      ) : (
        <div className="run-list">
          {runs.map((run) => (
            <article className="run-card" key={run.id}>
              <div className="paper-meta">
                <span>{run.model}</span>
                <span>{run.suite}</span>
              </div>
              <h3>{run.id}</h3>
              <div className="run-metrics">
                {Object.entries(run.metrics).map(([label, value]) => (
                  <span key={label}>
                    <strong>{metricLabel(value)}</strong>
                    {label}
                  </span>
                ))}
              </div>
              {run.artifactPaths.map((path) => (
                <a
                  className="artifact-link"
                  href={path}
                  key={path}
                  onClick={(event) => {
                    if (!isMarkdownHref(path)) {
                      return;
                    }
                    event.preventDefault();
                    onOpenMarkdown(path);
                  }}
                >
                  <ArrowUpRight aria-hidden="true" />
                  {basename(path)}
                </a>
              ))}
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
