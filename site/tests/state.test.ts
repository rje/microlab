import { describe, expect, it } from "vitest";

import { parseMicrolabState } from "../src/state";

describe("parseMicrolabState", () => {
  it("accepts valid phase, paper, synopsis, and eval run data", () => {
    const parsed = parseMicrolabState({
      phases: [
        {
          id: "phase-0",
          title: "Phase 0: Evaluation Harness",
          status: "current",
          goal: "Build evals first.",
          tasks: [
            {
              id: "schema",
              title: "Define schema",
              status: "active",
              why: "Stable records make later comparisons honest.",
              links: ["plans/phase-0.md"]
            }
          ],
          readingPaperIds: ["mmlu"]
        }
      ],
      papers: [
        {
          id: "mmlu",
          topic: "evaluation",
          title: "Measuring Massive Multitask Language Understanding",
          authors: "Hendrycks et al.",
          year: 2020,
          sourceUrl: "https://arxiv.org/abs/2009.03300",
          pdfUrl: "/papers/evaluation/mmlu.pdf",
          filename: "mmlu.pdf"
        }
      ],
      synopses: {
        mmlu: {
          paperId: "mmlu",
          oneSentence: "Broad multitask benchmark.",
          coreIdeas: ["Use many subjects."],
          whyItMatters: "It frames broad knowledge evaluation.",
          phaseConnection: "It informs the first suite.",
          suggestedReadingFocus: ["Subject design"]
        }
      },
      evalRuns: [
        {
          id: "phase0-smoke",
          phaseId: "phase-0",
          model: "fixture",
          suite: "smoke",
          createdAt: "2026-06-18T12:00:00Z",
          metrics: { passRate: 1 },
          artifactPaths: ["/artifacts/runs/evals/phase0-smoke/report.md"]
        }
      ]
    });

    expect(parsed.phases[0].tasks[0].status).toBe("active");
    expect(parsed.papers[0].pdfUrl).toContain("/papers/evaluation/");
    expect(parsed.synopses.mmlu.coreIdeas).toHaveLength(1);
    expect(parsed.evalRuns[0].metrics.passRate).toBe(1);
  });

  it("rejects malformed state before rendering", () => {
    expect(() => parseMicrolabState({ phases: [{ id: "phase-0" }] })).toThrow();
  });
});
