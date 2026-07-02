import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../src/App";
import type { MicrolabState } from "../src/state";

const state: MicrolabState = {
  phases: [
    {
      id: "phase-0",
      title: "Phase 0: Evaluation Harness",
      status: "current",
      goal: "Build reproducible evaluation before training.",
      summary: "Create the measurement layer for the lab.",
      tasks: [
        {
          id: "environment",
          title: "Environment and repo setup",
          status: "done",
          why: "Every experiment needs a stable base.",
          links: ["plans/environment-setup.md"]
        },
        {
          id: "eval-schema",
          title: "Evaluation schema and suite loader",
          status: "queued",
          why: "Stable task records make comparison possible.",
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
      oneSentence: "MMLU measures broad multitask knowledge.",
      summary:
        "MMLU is a broad multiple-choice benchmark that makes uneven model knowledge visible across subjects.",
      coreIdeas: ["Breadth matters.", "Aggregate scores need context."],
      whyItMatters: "It is the canonical broad knowledge eval.",
      phaseConnection: "It informs category-level Phase 0 reports.",
      suggestedReadingFocus: ["Subject construction", "Prompt settings"]
    }
  },
  evalRuns: [],
  csrfToken: "tok"
};

describe("App", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("renders the phase dashboard with a paper launcher", () => {
    render(<App initialState={state} />);

    expect(screen.getByRole("heading", { name: "Phase 0: Evaluation Harness" })).toBeInTheDocument();
    expect(screen.getByText("Environment and repo setup")).toBeInTheDocument();
    expect(screen.getByText("Measuring Massive Multitask Language Understanding")).toBeInTheDocument();
    expect(screen.getByText(/MMLU measures broad multitask knowledge/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /read & take notes/i })).toBeInTheDocument();
    expect(screen.getByText("Awaiting first eval run")).toBeInTheDocument();
  });

  it("opens markdown links in a rendered document viewer", async () => {
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({
        path: "plans/environment-setup.md",
        title: "Environment Setup",
        content: "# Environment Setup\n\n- Use the `microlab` conda environment.\n\n```bash\nconda run -n microlab pytest\n```"
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("link", { name: /environment-setup\.md/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/markdown?path=plans%2Fenvironment-setup.md"
      );
    });
    expect(
      await screen.findByRole("heading", { name: "Environment Setup" })
    ).toBeInTheDocument();
    expect(screen.getByText(/Use the.*conda environment/i)).toBeInTheDocument();
    expect(screen.getByText("microlab")).toBeInTheDocument();
    expect(screen.getByText("conda run -n microlab pytest")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: /raw markdown/i })).toHaveAttribute(
      "href",
      "/plans/environment-setup.md"
    );
  });

  it("opens the reading workspace and saves read-state from it", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      String(url).includes("/notes")
        ? { ok: true, json: async () => ({ paperId: "mmlu", content: "" }), text: async () => "" }
        : { ok: true, json: async () => ({}), text: async () => "" }
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /read & take notes/i }));

    expect(await screen.findByRole("dialog")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText(/^reading state$/i), { target: { value: "mapped" } });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/papers/mmlu/progress",
        expect.objectContaining({
          method: "POST",
          headers: expect.objectContaining({ "X-CSRF-Token": "tok" })
        })
      );
    });
  });

  it("loads existing notes into the workspace editor", async () => {
    const fetchMock = vi.fn(async (url: string) =>
      String(url).includes("/notes")
        ? {
            ok: true,
            json: async () => ({ paperId: "mmlu", content: "loaded note text" }),
            text: async () => ""
          }
        : { ok: true, json: async () => ({}), text: async () => "" }
    );
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /read & take notes/i }));
    expect(await screen.findByDisplayValue("loaded note text")).toBeInTheDocument();
  });

  it("changes a task status from the board and persists it", async () => {
    const fetchMock = vi.fn(async () => ({ ok: true, json: async () => ({}), text: async () => "" }));
    vi.stubGlobal("fetch", fetchMock);
    render(<App initialState={state} />);
    const select = screen.getByLabelText(/status for Evaluation schema and suite loader/i);
    fireEvent.change(select, { target: { value: "done" } });
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/phases/phase-0/tasks/eval-schema/status",
        expect.objectContaining({ method: "POST", headers: expect.objectContaining({ "X-CSRF-Token": "tok" }) })
      );
    });
  });

  it("shows the TensorBoard iframe when the Training tab is selected", () => {
    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /training/i }));
    const frame = screen.getByTitle("TensorBoard");
    expect(frame).toBeInTheDocument();
    expect(frame).toHaveAttribute("src", "/tensorboard/");
  });

  it("renders the AI overview in the summary tab when present", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url).includes("/overview")) {
        return {
          ok: true,
          json: async () => ({
            paperId: "mmlu",
            tldr: "Overview TL;DR line.",
            sections: [{ title: "1. Intro", summary: "Intro summary." }],
            readingFocus: []
          }),
          text: async () => ""
        };
      }
      if (String(url).includes("/notes")) {
        return { ok: true, json: async () => ({ paperId: "mmlu", content: "" }), text: async () => "" };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /read & take notes/i }));
    expect(await screen.findByText("Overview TL;DR line.")).toBeInTheDocument();
    expect(await screen.findByText("1. Intro")).toBeInTheDocument();
  });

  it("populates the playground run picker and reloads a checkpoint", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === "/api/serve/runs") {
        return {
          ok: true,
          json: async () => ({
            runs: [
              { name: "150m", latest_step: 6000 },
              { name: "350m", latest_step: 4000 }
            ],
            active: null
          })
        };
      }
      if (String(url) === "/api/serve/reload") {
        return { ok: true, json: async () => ({ run: "150m", step: 6200 }) };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /playground/i }));

    // The dropdown is populated from /api/serve/runs and defaults to the first run.
    const select = (await screen.findByRole("combobox", {
      name: /run to serve/i
    })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("150m"));
    expect(screen.getByRole("option", { name: /150m · step 6000/i })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /350m · step 4000/i })).toBeInTheDocument();

    // Reload latest posts to /api/serve/reload and reflects the fresh step as resident.
    fireEvent.click(screen.getByRole("button", { name: /reload latest/i }));
    await waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/serve/reload",
        expect.objectContaining({ method: "POST" })
      )
    );
    expect(
      await screen.findByText(/Serving 150m · step 6200 · resident/i)
    ).toBeInTheDocument();
  });
});
