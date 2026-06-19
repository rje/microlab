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

const paperSchema = z.object({
  id: z.string(),
  topic: z.string(),
  title: z.string(),
  authors: z.string(),
  year: z.number(),
  sourceUrl: z.string().url(),
  pdfUrl: z.string(),
  filename: z.string()
});

const paperSynopsisSchema = z.object({
  paperId: z.string(),
  oneSentence: z.string(),
  coreIdeas: z.array(z.string()),
  whyItMatters: z.string(),
  phaseConnection: z.string(),
  suggestedReadingFocus: z.array(z.string())
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

const microlabStateSchema = z.object({
  phases: z.array(phaseSchema),
  papers: z.array(paperSchema),
  synopses: z.record(paperSynopsisSchema),
  evalRuns: z.array(evalRunSummarySchema)
});

export type PhaseTask = z.infer<typeof phaseTaskSchema>;
export type Phase = z.infer<typeof phaseSchema>;
export type Paper = z.infer<typeof paperSchema>;
export type PaperSynopsis = z.infer<typeof paperSynopsisSchema>;
export type EvalRunSummary = z.infer<typeof evalRunSummarySchema>;
export type MicrolabState = z.infer<typeof microlabStateSchema>;

export function parseMicrolabState(value: unknown): MicrolabState {
  return microlabStateSchema.parse(value);
}

export async function fetchMicrolabState(): Promise<MicrolabState> {
  const response = await fetch("/api/state");
  if (!response.ok) {
    throw new Error(`Failed to load state: ${response.status}`);
  }
  return parseMicrolabState(await response.json());
}
