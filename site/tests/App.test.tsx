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

  it("frames a chat run as a message exchange and can force raw completion", async () => {
    const generateCalls: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url) === "/api/serve/runs") {
        return {
          ok: true,
          json: async () => ({
            runs: [
              { name: "350m", latest_step: 4000, mode: "base" },
              { name: "350m-sft", latest_step: 900, mode: "chat" }
            ],
            active: null
          })
        };
      }
      if (String(url) === "/api/generate") {
        generateCalls.push(JSON.parse(String(init?.body)));
        // Minimal streaming body: one empty read then done.
        return {
          ok: true,
          headers: new Headers(),
          body: { getReader: () => ({ read: async () => ({ done: true, value: undefined }) }) }
        };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /playground/i }));

    // Base run first: no raw toggle, prompt is a "Prompt".
    const select = (await screen.findByRole("combobox", { name: /run to serve/i })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("350m"));
    expect(screen.queryByRole("checkbox", { name: /raw completion/i })).not.toBeInTheDocument();

    // Switch to the chat run: message framing + raw-completion toggle appear.
    fireEvent.change(select, { target: { value: "350m-sft" } });
    expect(await screen.findByLabelText(/message/i)).toBeInTheDocument();
    const rawToggle = screen.getByRole("checkbox", { name: /raw completion/i });
    expect(screen.getByRole("button", { name: /^send$/i })).toBeInTheDocument();

    // Send as chat -> raw:false.
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));
    await waitFor(() => expect(generateCalls).toHaveLength(1));
    expect(generateCalls[0]).toMatchObject({ run: "350m-sft", raw: false });

    // Toggle raw completion -> the next request forces raw:true.
    fireEvent.click(rawToggle);
    fireEvent.click(screen.getByRole("button", { name: /generate/i }));
    await waitFor(() => expect(generateCalls).toHaveLength(2));
    expect(generateCalls[1]).toMatchObject({ run: "350m-sft", raw: true });
  });

  // One chunk of streamed text, then done — enough for the reader loop to accumulate a reply.
  const streamBody = (text: string) => ({
    getReader: () => {
      let sent = false;
      return {
        read: async () => {
          if (sent) return { done: true, value: undefined };
          sent = true;
          return { done: false, value: new TextEncoder().encode(text) };
        }
      };
    }
  });

  const chatRunsResponse = {
    ok: true,
    json: async () => ({
      runs: [{ name: "350m-sft", latest_step: 900, mode: "chat" }],
      active: null
    })
  };

  const openChatPlayground = async () => {
    render(<App initialState={state} />);
    fireEvent.click(screen.getByRole("button", { name: /playground/i }));
    const select = (await screen.findByRole("combobox", {
      name: /run to serve/i
    })) as HTMLSelectElement;
    await waitFor(() => expect(select.value).toBe("350m-sft"));
  };

  const sendMessage = async (text: string) => {
    fireEvent.change(screen.getByLabelText(/message/i), { target: { value: text } });
    fireEvent.click(screen.getByRole("button", { name: /^send$/i }));
    // The send is complete (and the exchange committed) once the button reads "Send" again.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^send$/i })).toBeEnabled()
    );
  };

  it("sends the visible transcript as history and clears it on demand", async () => {
    const generateCalls: Array<Record<string, unknown>> = [];
    const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url) === "/api/serve/runs") return chatRunsResponse;
      if (String(url) === "/api/generate") {
        generateCalls.push(JSON.parse(String(init?.body)));
        return {
          ok: true,
          headers: new Headers({ "X-Chat-Turns-Used": "0", "X-Chat-Turns-Dropped": "0" }),
          body: streamBody(`reply ${generateCalls.length}`)
        };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    await openChatPlayground();

    // First send: empty history; the exchange lands in the transcript.
    await sendMessage("hi model");
    expect(generateCalls[0]).toMatchObject({ prompt: "hi model", history: [] });
    expect(screen.getByText("hi model")).toBeInTheDocument();
    expect(screen.getByText("reply 1")).toBeInTheDocument();

    // Follow-up: the completed exchange rides along as history.
    await sendMessage("tell me more");
    expect(generateCalls[1]).toMatchObject({
      prompt: "tell me more",
      history: [{ user: "hi model", assistant: "reply 1" }]
    });
    expect(screen.getByText("reply 2")).toBeInTheDocument();

    // Clear conversation: transcript is emptied and the next send starts fresh.
    fireEvent.click(screen.getByRole("button", { name: /clear conversation/i }));
    expect(screen.queryByText("reply 1")).not.toBeInTheDocument();
    expect(screen.queryByText("reply 2")).not.toBeInTheDocument();
    await sendMessage("fresh start");
    expect(generateCalls[2]).toMatchObject({ prompt: "fresh start", history: [] });
  });

  it("shows a dropped-turns indicator when the server trims the conversation", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === "/api/serve/runs") return chatRunsResponse;
      if (String(url) === "/api/generate") {
        return {
          ok: true,
          headers: new Headers({ "X-Chat-Turns-Used": "1", "X-Chat-Turns-Dropped": "2" }),
          body: streamBody("trimmed reply")
        };
      }
      return { ok: true, json: async () => ({}), text: async () => "" };
    });
    vi.stubGlobal("fetch", fetchMock);

    await openChatPlayground();
    expect(screen.queryByText(/dropped the oldest/i)).not.toBeInTheDocument();
    await sendMessage("hello");
    expect(await screen.findByText(/dropped the oldest 2 turns/i)).toBeInTheDocument();
  });
});
