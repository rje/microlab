import { z } from "zod";

const phaseTaskSchema = z.object({
  id: z.string(),
  title: z.string(),
  status: z.enum(["done", "active", "queued", "blocked"]),
  why: z.string(),
  links: z.array(z.string()).default([])
});

const phaseSchema = z.object({
  id: z.string(),
  title: z.string(),
  status: z.enum(["current", "planned", "complete"]),
  goal: z.string(),
  summary: z.string().default(""),
  tasks: z.array(phaseTaskSchema),
  readingPaperIds: z.array(z.string())
});

const paperProgressSchema = z.object({
  readState: z.string(),
  depth: z.string().nullable()
});

const paperSchema = z.object({
  id: z.string(),
  topic: z.string(),
  title: z.string(),
  authors: z.string(),
  year: z.number(),
  sourceUrl: z.string().url(),
  pdfUrl: z.string(),
  filename: z.string(),
  progress: paperProgressSchema.optional()
});

const paperSynopsisSchema = z.object({
  paperId: z.string(),
  oneSentence: z.string(),
  summary: z.string(),
  coreIdeas: z.array(z.string()),
  whyItMatters: z.string(),
  phaseConnection: z.string(),
  suggestedReadingFocus: z.array(z.string())
});

const paperOverviewSchema = z.object({
  paperId: z.string(),
  generatedAt: z.string().optional(),
  depthSuggestion: z.string().optional(),
  tldr: z.string(),
  overview: z.string().optional(),
  sections: z.array(z.object({ title: z.string(), summary: z.string() })).default([]),
  readingFocus: z.array(z.string()).default([])
});

const evalRunSummarySchema = z.object({
  id: z.string(),
  phaseId: z.string(),
  model: z.string(),
  suite: z.string(),
  createdAt: z.string(),
  metrics: z.record(z.number()),
  artifactPaths: z.array(z.string())
});

const markdownDocumentSchema = z.object({
  path: z.string(),
  title: z.string(),
  content: z.string()
});

const microlabStateSchema = z.object({
  phases: z.array(phaseSchema),
  papers: z.array(paperSchema),
  synopses: z.record(paperSynopsisSchema),
  evalRuns: z.array(evalRunSummarySchema),
  csrfToken: z.string().optional()
});

export type PhaseTask = z.infer<typeof phaseTaskSchema>;
export type Phase = z.infer<typeof phaseSchema>;
export type Paper = z.infer<typeof paperSchema>;
export type PaperProgress = z.infer<typeof paperProgressSchema>;
export type PaperSynopsis = z.infer<typeof paperSynopsisSchema>;
export type PaperOverview = z.infer<typeof paperOverviewSchema>;
export type EvalRunSummary = z.infer<typeof evalRunSummarySchema>;
export type MarkdownDocument = z.infer<typeof markdownDocumentSchema>;
export type MicrolabState = z.infer<typeof microlabStateSchema>;

export function parseMicrolabState(value: unknown): MicrolabState {
  return microlabStateSchema.parse(value);
}

export async function fetchMicrolabState(): Promise<MicrolabState> {
  const response = await fetch("/api/state");
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail.trim() || `Failed to load state: ${response.status}`);
  }
  return parseMicrolabState(await response.json());
}

export function parseMarkdownDocument(value: unknown): MarkdownDocument {
  return markdownDocumentSchema.parse(value);
}

export async function fetchMarkdownDocument(path: string): Promise<MarkdownDocument> {
  const response = await fetch(`/api/markdown?path=${encodeURIComponent(path)}`);
  if (!response.ok) {
    throw new Error(`Failed to load markdown: ${response.status}`);
  }
  return parseMarkdownDocument(await response.json());
}

const notesSchema = z.object({ paperId: z.string(), content: z.string() });

async function mutate(path: string, csrfToken: string, body: unknown): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
    body: JSON.stringify(body)
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail.trim() || `Request failed: ${response.status}`);
  }
}

export function saveProgress(
  paperId: string,
  csrfToken: string,
  progress: { readState: string; depth: string | null }
): Promise<void> {
  return mutate(`/api/papers/${encodeURIComponent(paperId)}/progress`, csrfToken, progress);
}

export function saveNotes(paperId: string, csrfToken: string, content: string): Promise<void> {
  return mutate(`/api/papers/${encodeURIComponent(paperId)}/notes`, csrfToken, { content });
}

export function saveTaskStatus(
  phaseId: string,
  taskId: string,
  csrfToken: string,
  status: string
): Promise<void> {
  return mutate(
    `/api/phases/${encodeURIComponent(phaseId)}/tasks/${encodeURIComponent(taskId)}/status`,
    csrfToken,
    { status }
  );
}

export async function fetchNotes(paperId: string): Promise<string> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/notes`);
  if (!response.ok) {
    throw new Error(`Failed to load notes: ${response.status}`);
  }
  return notesSchema.parse(await response.json()).content;
}

export async function fetchOverview(paperId: string): Promise<PaperOverview | null> {
  try {
    const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/overview`);
    if (!response.ok) {
      return null;
    }
    return paperOverviewSchema.parse(await response.json());
  } catch {
    return null;
  }
}

export type DueCard = {
  id: string;
  paperId: string;
  paperTitle: string;
  question: string;
  answer: string;
  status: string;
};

export async function fetchDueCards(): Promise<{ cards: DueCard[]; total: number }> {
  const response = await fetch("/api/recall/due");
  if (!response.ok) {
    throw new Error(`Failed to load due cards: ${response.status}`);
  }
  return response.json();
}

export function submitReview(
  cardId: string,
  paperId: string,
  csrfToken: string,
  grade: number
): Promise<void> {
  return mutate("/api/recall/review", csrfToken, { cardId, paperId, grade });
}

export type Highlight = {
  id: number;
  page: number;
  rects: { x: number; y: number; w: number; h: number }[];
  text: string;
};

export async function fetchHighlights(paperId: string): Promise<Highlight[]> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/highlights`);
  if (!response.ok) {
    throw new Error(`Failed to load highlights: ${response.status}`);
  }
  return (await response.json()).highlights ?? [];
}

export async function addHighlight(
  paperId: string,
  csrfToken: string,
  highlight: { page: number; rects: { x: number; y: number; w: number; h: number }[]; text: string }
): Promise<Highlight> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/highlights`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": csrfToken },
    body: JSON.stringify(highlight)
  });
  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new Error(detail.trim() || `Save failed: ${response.status}`);
  }
  return response.json();
}

export async function deleteHighlight(
  paperId: string,
  csrfToken: string,
  id: number
): Promise<void> {
  const response = await fetch(`/api/papers/${encodeURIComponent(paperId)}/highlights/${id}`, {
    method: "DELETE",
    headers: { "X-CSRF-Token": csrfToken }
  });
  if (!response.ok) {
    throw new Error(`Delete failed: ${response.status}`);
  }
}
