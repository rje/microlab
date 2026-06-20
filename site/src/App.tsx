import {
  Activity,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  Circle,
  Clock3,
  ExternalLink,
  FileText,
  FlaskConical,
  Layers3,
  PlayCircle,
  TerminalSquare
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import {
  EvalRunSummary,
  MarkdownDocument,
  MicrolabState,
  Paper,
  PaperSynopsis,
  Phase,
  PhaseTask,
  fetchMarkdownDocument,
  fetchMicrolabState,
  fetchNotes,
  saveNotes,
  saveProgress
} from "./state";
import { MarkdownDocumentView } from "./MarkdownDocumentView";

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
          <TaskBoard tasks={activePhase.tasks} onOpenMarkdown={openMarkdownDocument} />
        </main>
        <aside className="right-rail">
          <ReadingPanel
            papers={readingPapers}
            synopses={state.synopses}
            csrfToken={state.csrfToken ?? ""}
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
  onOpenMarkdown
}: {
  tasks: PhaseTask[];
  onOpenMarkdown: (href: string) => void;
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
  csrfToken
}: {
  papers: Paper[];
  synopses: MicrolabState["synopses"];
  csrfToken: string;
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
          <PaperCard
            key={paper.id}
            paper={paper}
            synopsis={synopses[paper.id]}
            csrfToken={csrfToken}
          />
        ))}
      </div>
    </section>
  );
}

function PaperCard({
  paper,
  synopsis,
  csrfToken
}: {
  paper: Paper;
  synopsis: PaperSynopsis | undefined;
  csrfToken: string;
}) {
  const [readState, setReadState] = useState(paper.progress?.readState ?? "unread");
  const [depth, setDepth] = useState<string>(paper.progress?.depth ?? "");
  const [notesOpen, setNotesOpen] = useState(false);
  const [notes, setNotes] = useState("");
  const [notesLoaded, setNotesLoaded] = useState(false);
  const [saved, setSaved] = useState("");

  const persist = (next: { readState?: string; depth?: string }) => {
    const rs = next.readState ?? readState;
    const d = next.depth ?? depth;
    saveProgress(paper.id, csrfToken, { readState: rs, depth: d === "" ? null : d })
      .then(() => setSaved("saved"))
      .catch((error: Error) => setSaved(error.message));
  };

  const toggleNotes = async () => {
    setNotesOpen((open) => !open);
    if (!notesLoaded) {
      try {
        setNotes(await fetchNotes(paper.id));
      } catch {
        setNotes("");
      }
      setNotesLoaded(true);
    }
  };

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

  return (
    <article className="paper-card">
      <div className="paper-meta">
        <span>{paper.topic}</span>
        <span>{paper.year}</span>
      </div>
      <h3>{paper.title}</h3>
      <p className="authors">{paper.authors}</p>

      <div className="progress-controls">
        <select
          aria-label={`Reading state for ${paper.title}`}
          value={readState}
          onChange={(event) => {
            setReadState(event.target.value);
            persist({ readState: event.target.value });
          }}
        >
          {READ_STATES.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <select
          aria-label={`Depth for ${paper.title}`}
          value={depth}
          onChange={(event) => {
            setDepth(event.target.value);
            persist({ depth: event.target.value });
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
      </div>

      {synopsis && (
        <>
          <p className="synopsis-lede">{synopsis.oneSentence}</p>
          <div className="study-notes">
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
              <strong>Phase connection</strong>
              <p>{synopsis.phaseConnection}</p>
            </div>
            <div className="study-block">
              <strong>Reading focus</strong>
              <ul className="study-list">
                {synopsis.suggestedReadingFocus.map((focus) => (
                  <li key={focus}>{focus}</li>
                ))}
              </ul>
            </div>
          </div>
        </>
      )}

      <div className="paper-actions">
        <button type="button" onClick={toggleNotes}>
          <FileText aria-hidden="true" />
          {notesOpen ? "Hide notes" : "Notes"}
        </button>
        <a href={paper.pdfUrl}>
          <FileText aria-hidden="true" />
          PDF
        </a>
        <a href={paper.sourceUrl} rel="noreferrer" target="_blank">
          <ExternalLink aria-hidden="true" />
          Source
        </a>
      </div>

      {notesOpen && (
        <textarea
          className="notes-editor"
          aria-label={`Notes for ${paper.title}`}
          value={notes}
          placeholder="Your notes (mechanism in your own words, what not to copy, questions)…"
          onChange={(event) => setNotes(event.target.value)}
          rows={8}
        />
      )}
    </article>
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
