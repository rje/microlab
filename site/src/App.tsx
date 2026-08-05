import {
  Activity,
  LineChart,
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
  Sparkles,
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
  const [view, setView] = useState<"phases" | "training" | "playground" | "runlog">("phases");

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
          activeView={view}
          onSelectPhase={(phaseId) => {
            setActivePhaseId(phaseId);
            setView("phases");
          }}
          onSelectTraining={() => setView("training")}
          onSelectPlayground={() => setView("playground")}
          onSelectRunLog={() => setView("runlog")}
        />
        {view === "training" ? (
          <main className="workspace workspace-full">
            <TrainingPanel />
          </main>
        ) : view === "playground" ? (
          <main className="workspace workspace-full">
            <PlaygroundPanel />
          </main>
        ) : view === "runlog" ? (
          <main className="workspace workspace-full">
            <RunLogPanel onOpenMarkdown={openMarkdownDocument} />
          </main>
        ) : (
          <>
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
          </>
        )}
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
  activeView,
  onSelectPhase,
  onSelectTraining,
  onSelectPlayground,
  onSelectRunLog
}: {
  activePhaseId: string;
  phases: Phase[];
  activeView: "phases" | "training" | "playground" | "runlog";
  onSelectPhase: (phaseId: string) => void;
  onSelectTraining: () => void;
  onSelectPlayground: () => void;
  onSelectRunLog: () => void;
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
            className={`phase-nav ${
              activeView === "phases" && phase.id === activePhaseId ? "is-active" : ""
            }`}
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
        <button
          className={`phase-nav ${activeView === "training" ? "is-active" : ""}`}
          onClick={onSelectTraining}
          type="button"
        >
          <span className="phase-number" aria-hidden="true">
            <Activity />
          </span>
          <span>
            <strong>Training</strong>
            <small>TensorBoard</small>
          </span>
        </button>
        <button
          className={`phase-nav ${activeView === "runlog" ? "is-active" : ""}`}
          onClick={onSelectRunLog}
          type="button"
        >
          <span className="phase-number" aria-hidden="true">
            <LineChart />
          </span>
          <span>
            <strong>Run log</strong>
            <small>Milestones &amp; sweeps</small>
          </span>
        </button>
        <button
          className={`phase-nav ${activeView === "playground" ? "is-active" : ""}`}
          onClick={onSelectPlayground}
          type="button"
        >
          <span className="phase-number" aria-hidden="true">
            <Sparkles />
          </span>
          <span>
            <strong>Playground</strong>
            <small>Your model, live</small>
          </span>
        </button>
      </div>
    </nav>
  );
}

type TrajectoryRun = {
  run: string;
  steps: number[];
  completions: Record<string, Record<string, string>>;
  docs: string[];
};
type TrajectoryPrompt = { id: string; text: string };

function RunLogPanel({ onOpenMarkdown }: { onOpenMarkdown: (href: string) => void }) {
  const [runs, setRuns] = useState<TrajectoryRun[]>([]);
  const [prompts, setPrompts] = useState<TrajectoryPrompt[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Open steps per (run, prompt) — a SET, not a single selection: the whole point of the
  // trajectory is comparing checkpoints side by side (2k next to 4k next to 10k).
  const [open, setOpen] = useState<Record<string, number[]>>({});

  useEffect(() => {
    fetch("/api/trajectory")
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status}`))))
      .then((d) => {
        setRuns(d.runs ?? []);
        setPrompts(d.prompts ?? []);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const toggle = (key: string, step: number) =>
    setOpen((o) => {
      const cur = o[key] ?? [];
      return {
        ...o,
        [key]: cur.includes(step) ? cur.filter((s) => s !== step) : [...cur, step].sort((a, b) => a - b)
      };
    });

  return (
    <section className="training-panel" aria-labelledby="runlog-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Trajectory</p>
          <h2 id="runlog-heading">Run log</h2>
        </div>
        <LineChart aria-hidden="true" />
      </div>
      {error && <p role="alert">Failed to load trajectory: {error}</p>}
      {runs.map((r) => (
        <article key={r.run} style={{ marginBottom: "2rem" }}>
          <h3>{r.run}</h3>
          <p>
            {r.docs.map((d) => (
              <button
                className="phase-nav"
                key={d}
                onClick={() => onOpenMarkdown(d)}
                style={{ display: "inline-flex", width: "auto", marginRight: "0.5rem" }}
                type="button"
              >
                <FileText aria-hidden="true" />
                <span>{d.replace(/^docs\//, "").replace(/\.md$/, "")}</span>
              </button>
            ))}
          </p>
          {prompts
            .filter((q) => r.completions[q.id])
            .map((q) => {
              const key = `${r.run}:${q.id}`;
              const openSteps = (open[key] ?? []).filter(
                (s) => r.completions[q.id][String(s)] !== undefined
              );
              return (
                <details key={q.id} style={{ marginBottom: "0.75rem" }}>
                  <summary>
                    <code>{q.id}</code>
                  </summary>
                  <pre style={{ whiteSpace: "pre-wrap" }}>{q.text}</pre>
                  <p>
                    {r.steps
                      .filter((s) => r.completions[q.id][String(s)] !== undefined)
                      .map((s) => (
                        <button
                          className={`phase-nav ${openSteps.includes(s) ? "is-active" : ""}`}
                          key={s}
                          onClick={() => toggle(key, s)}
                          style={{ display: "inline-flex", width: "auto", marginRight: "0.4rem" }}
                          type="button"
                        >
                          step {s}
                        </button>
                      ))}
                  </p>
                  {openSteps.length > 0 && (
                    <div
                      style={{
                        display: "grid",
                        gap: "0.75rem",
                        gridTemplateColumns: `repeat(${Math.min(openSteps.length, 4)}, minmax(0, 1fr))`,
                        overflowX: "auto"
                      }}
                    >
                      {openSteps.map((s) => (
                        <div key={s} style={{ minWidth: 0 }}>
                          <p className="eyebrow">step {s}</p>
                          <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>
                            {r.completions[q.id][String(s)]}
                          </pre>
                        </div>
                      ))}
                    </div>
                  )}
                </details>
              );
            })}
        </article>
      ))}
    </section>
  );
}

function TrainingPanel() {
  return (
    <section className="training-panel" aria-labelledby="training-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Live</p>
          <h2 id="training-heading">Training</h2>
        </div>
        <Activity aria-hidden="true" />
      </div>
      <iframe
        title="TensorBoard"
        src="/tensorboard/"
        style={{ width: "100%", border: 0, height: "calc(100vh - 96px)" }}
      />
    </section>
  );
}

type Decoding = { temperature: number; top_p: number; top_k: number; repetition_penalty: number };
type ServeRun = {
  name: string;
  latestStep: number;
  mode: "chat" | "base";
  decoding: Decoding;
};
const OFF_DECODING: Decoding = { temperature: 0.8, top_p: 0, top_k: 0, repetition_penalty: 1.0 };

type Exchange = { user: string; assistant: string };

function PlaygroundPanel() {
  const [prompt, setPrompt] = useState("Once upon a time");
  const [output, setOutput] = useState("");
  // Completed (user, assistant) exchanges of the visible transcript. Sent as `history` with
  // every chat send so a multi-turn model conditions on the whole conversation.
  const [exchanges, setExchanges] = useState<Exchange[]>([]);
  // The message currently being answered (shown in the thread while the reply streams).
  const [pendingUser, setPendingUser] = useState<string | null>(null);
  // How many leading turns the server dropped to fit the context window (from the
  // X-Chat-Turns-Dropped response header); > 0 shows the truncation notice.
  const [droppedTurns, setDroppedTurns] = useState(0);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(0); // 0 = off
  const [topP, setTopP] = useState(0); // 0 = off
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0); // 1.0 = off
  const [maxTokens, setMaxTokens] = useState(512);
  const [seed, setSeed] = useState(""); // blank = no seed (fresh samples each run)
  // Chat runs answer a message; "raw completion" forces base-style raw output (raw:true) on a
  // chat model so you can compare the instruction-tuned reply against the untemplated model.
  const [rawMode, setRawMode] = useState(false);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Run/checkpoint picker: the server keeps ONE model resident, so selecting a run (or
  // reloading) switches which trained run's LATEST checkpoint answers /api/generate.
  const [runs, setRuns] = useState<ServeRun[]>([]);
  const [selectedRun, setSelectedRun] = useState(""); // "" = let the server pick its default
  const [activeRun, setActiveRun] = useState<string | null>(null); // resident on the server
  const [activeStep, setActiveStep] = useState<number | null>(null);
  const [reloading, setReloading] = useState(false);

  // Pre-fill the sampling sliders from a run's per-model defaults when it's selected. Only fired
  // on an explicit run change / initial load, so it never clobbers the user's manual tweaks.
  const applyRunDecoding = (run: ServeRun | undefined) => {
    if (!run) return;
    setTemperature(run.decoding.temperature);
    setTopP(run.decoding.top_p);
    setTopK(run.decoding.top_k);
    setRepetitionPenalty(run.decoding.repetition_penalty);
  };

  // Kill an in-flight generation when navigating away — the server holds a single-generation
  // lock, so a leaked stream would block the next request.
  useEffect(() => () => abortRef.current?.abort(), []);

  // Populate the picker from the server's runs. Failures are non-fatal: the Playground still
  // works against the server's default run, just without the switcher.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/serve/runs");
        if (!res.ok) return;
        const data = await res.json();
        if (cancelled) return;
        const list: ServeRun[] = (data.runs ?? []).map(
          (r: {
            name: string;
            latest_step: number;
            mode?: string;
            decoding?: Decoding;
          }) => ({
            name: r.name,
            latestStep: r.latest_step,
            mode: r.mode === "chat" ? "chat" : "base",
            decoding: r.decoding ?? OFF_DECODING
          })
        );
        setRuns(list);
        const active = data.active as { name: string; step: number } | null;
        if (active) {
          setActiveRun(active.name);
          setActiveStep(active.step);
          setSelectedRun(active.name);
          applyRunDecoding(list.find((r) => r.name === active.name));
        } else if (list.length) {
          setSelectedRun(list[0].name);
          applyRunDecoding(list[0]);
        }
      } catch {
        /* leave the picker empty; generate still hits the default run */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const reload = async () => {
    setReloading(true);
    setError(null);
    try {
      const res = await fetch("/api/serve/reload", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selectedRun ? { run: selectedRun } : {})
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error ?? `HTTP ${res.status}`);
      setActiveRun(data.run);
      setActiveStep(data.step);
      setSelectedRun(data.run);
      setRuns((prev) =>
        prev.map((r) => (r.name === data.run ? { ...r, latestStep: data.step } : r))
      );
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setReloading(false);
    }
  };

  const selected = runs.find((r) => r.name === selectedRun) ?? null;
  const isResident = activeRun !== null && activeRun === selectedRun;
  const shownStep = isResident ? activeStep : selected?.latestStep ?? null;
  const isChat = selected?.mode === "chat";
  // A chat run answers as itself unless "raw completion" is on; base runs are always raw.
  const chatReply = isChat && !rawMode;

  // Fold the streamed (or stopped-early) reply into the transcript. An effectively empty
  // reply is NOT committed — the backend rejects empty assistant turns in history, so the
  // message goes back into the box for a retry instead of poisoning the conversation.
  const commitExchange = (user: string, assistant: string) => {
    setPendingUser(null);
    setOutput("");
    if (assistant.trim()) {
      setExchanges((prev) => [...prev, { user, assistant }]);
    } else {
      setPrompt(user);
    }
  };

  const clearConversation = () => {
    setExchanges([]);
    setPendingUser(null);
    setOutput("");
    setDroppedTurns(0);
    setStats(null);
    setError(null);
  };

  const generate = async () => {
    const message = prompt;
    const asChat = chatReply; // pin the mode for this send; toggles mid-stream can't skew it
    setRunning(true);
    setOutput("");
    setStats(null);
    setError(null);
    if (asChat) {
      setPendingUser(message);
      setPrompt("");
    }
    const controller = new AbortController();
    abortRef.current = controller;
    const t0 = performance.now();
    let reply = "";
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: message,
          max_new_tokens: maxTokens,
          temperature,
          top_k: topK || null,
          top_p: topP || null,
          repetition_penalty: repetitionPenalty,
          seed: seed === "" ? null : Number(seed),
          run: selectedRun || null,
          raw: isChat && rawMode,
          // The visible transcript rides along on chat sends so the model sees the whole
          // conversation; base runs and raw completions stay single-shot.
          ...(asChat ? { history: exchanges } : {})
        }),
        signal: controller.signal
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      if (asChat) {
        setDroppedTurns(Number(res.headers.get("X-Chat-Turns-Dropped") ?? "0"));
      }
      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        const piece = decoder.decode(value, { stream: true });
        reply += piece;
        setOutput((prev) => prev + piece);
      }
      // A successful generate loaded the selected run's latest checkpoint into residency.
      if (selectedRun) {
        setActiveRun(selectedRun);
        if (selected) setActiveStep(selected.latestStep);
      }
      const secs = (performance.now() - t0) / 1000;
      setStats(
        `${(reply.length / Math.max(secs, 0.001)).toFixed(0)} chars/s · ${secs.toFixed(1)}s`
      );
      if (asChat) commitExchange(message, reply);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        // Stopped mid-reply: what streamed still happened, so it stays in the transcript.
        if (asChat) commitExchange(message, reply);
      } else {
        setError((err as Error).message);
        if (asChat) {
          setPendingUser(null);
          setPrompt(message); // failed send: put the message back for a retry
        }
      }
    } finally {
      setRunning(false);
    }
  };

  return (
    <section className="playground-panel" aria-labelledby="playground-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Live</p>
          <h2 id="playground-heading">Playground</h2>
        </div>
        <Sparkles aria-hidden="true" />
      </div>

      <div className="playground-serving">
        <label className="playground-serving-run">
          <span className="playground-label">Run</span>
          <select
            className="playground-serving-select"
            aria-label="Run to serve"
            value={selectedRun}
            disabled={runs.length === 0}
            onChange={(event) => {
              setSelectedRun(event.target.value);
              applyRunDecoding(runs.find((r) => r.name === event.target.value));
            }}
          >
            {runs.length === 0 && <option value="">default run</option>}
            {runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} · step {r.latestStep}
              </option>
            ))}
          </select>
        </label>
        <span className={`playground-mode-badge is-${isChat ? "chat" : "base"}`}>
          {isChat ? "chat" : "base"}
        </span>
        {isChat && (
          <label className="playground-raw-toggle">
            <input
              type="checkbox"
              checked={rawMode}
              onChange={(event) => setRawMode(event.target.checked)}
            />
            <span>raw completion</span>
          </label>
        )}
        <button
          type="button"
          className="playground-reload"
          onClick={reload}
          disabled={reloading || running}
        >
          {reloading ? "Reloading…" : "Reload latest"}
        </button>
        <span className="playground-serving-status">
          {selectedRun
            ? `Serving ${selectedRun}${shownStep != null ? ` · step ${shownStep}` : ""}${
                isResident ? " · resident" : ""
              }`
            : "Serving the default run"}
        </span>
      </div>

      <div className="playground-controls">
        <label className="playground-control">
          <span className="playground-label">Temperature {temperature.toFixed(1)}</span>
          <input
            type="range"
            min={0}
            max={2}
            step={0.1}
            value={temperature}
            onChange={(event) => setTemperature(Number(event.target.value))}
          />
        </label>
        <label className="playground-control">
          <span className="playground-label">Top-p {topP.toFixed(2)} (0 = off)</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.05}
            value={topP}
            onChange={(event) => setTopP(Number(event.target.value))}
          />
        </label>
        <label className="playground-control">
          <span className="playground-label">
            Repetition penalty {repetitionPenalty.toFixed(2)} (1 = off)
          </span>
          <input
            type="range"
            min={1}
            max={2}
            step={0.05}
            value={repetitionPenalty}
            onChange={(event) => setRepetitionPenalty(Number(event.target.value))}
          />
        </label>
        <label className="playground-control">
          <span className="playground-label">Top-k (0 = off)</span>
          <input
            type="number"
            min={0}
            step={1}
            value={topK}
            onChange={(event) => setTopK(Number(event.target.value))}
          />
        </label>
        <label className="playground-control">
          <span className="playground-label">Max tokens</span>
          <input
            type="number"
            min={1}
            max={4096}
            step={1}
            value={maxTokens}
            onChange={(event) => setMaxTokens(Number(event.target.value))}
          />
        </label>
        <label className="playground-control">
          <span className="playground-label">Seed (blank = off)</span>
          <input
            type="number"
            step={1}
            value={seed}
            placeholder="none"
            onChange={(event) => setSeed(event.target.value)}
          />
        </label>
      </div>

      <label className="playground-field">
        <span className="playground-label">{chatReply ? "Your message" : "Prompt"}</span>
        <textarea
          className="playground-prompt"
          aria-label={chatReply ? "Message" : "Prompt"}
          rows={3}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
        />
      </label>

      <div className="playground-actions">
        <button
          type="button"
          className="playground-run"
          onClick={generate}
          disabled={running}
        >
          {chatReply ? (running ? "Sending…" : "Send") : running ? "Generating…" : "Generate"}
        </button>
        <button
          type="button"
          className="playground-stop"
          onClick={() => abortRef.current?.abort()}
          disabled={!running}
        >
          Stop
        </button>
        {chatReply && (
          <button
            type="button"
            className="playground-clear"
            onClick={clearConversation}
            disabled={running || (exchanges.length === 0 && pendingUser === null)}
          >
            Clear conversation
          </button>
        )}
        {stats && <span className="playground-stats">{stats}</span>}
      </div>

      {error && <p className="playground-error">{error}</p>}

      {chatReply && droppedTurns > 0 && (
        <p className="playground-truncated" role="status">
          Context window full — dropped the oldest {droppedTurns}{" "}
          {droppedTurns === 1 ? "turn" : "turns"} from the conversation.
        </p>
      )}

      {chatReply ? (
        <div className="playground-thread" aria-live="polite" aria-label="Conversation">
          {exchanges.length === 0 && pendingUser === null && (
            <p className="playground-thread-empty">
              No messages yet — the whole thread is sent with each message.
            </p>
          )}
          {exchanges.map((exchange, i) => (
            <div className="playground-exchange" key={i}>
              <div className="playground-msg is-user">
                <span className="playground-msg-role">You</span>
                <div className="playground-msg-text">{exchange.user}</div>
              </div>
              <div className="playground-msg is-assistant">
                <span className="playground-msg-role">{selectedRun || "model"}</span>
                <div className="playground-msg-text">{exchange.assistant}</div>
              </div>
            </div>
          ))}
          {pendingUser !== null && (
            <div className="playground-exchange">
              <div className="playground-msg is-user">
                <span className="playground-msg-role">You</span>
                <div className="playground-msg-text">{pendingUser}</div>
              </div>
              <div className="playground-msg is-assistant">
                <span className="playground-msg-role">{selectedRun || "model"}</span>
                <div className="playground-msg-text">{output || "…"}</div>
              </div>
            </div>
          )}
        </div>
      ) : (
        <pre className="playground-output" aria-live="polite">
          {output}
        </pre>
      )}
    </section>
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
