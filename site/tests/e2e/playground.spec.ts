import { expect, test } from "@playwright/test";

// The app tsconfig has no @types/node; declare the one Node global this spec reads.
declare const process: { env: Record<string, string | undefined> };

// Authed Playground e2e. The console requires login; we inject a session cookie signed with the
// app's local secret key (scripts/e2e_session_cookie.py) rather than the password, so this runs
// with no credential exchange. Skips cleanly when the cookie isn't provided.
const SESSION = process.env.MICROLAB_SESSION_COOKIE;

test.describe("Playground per-model decoding defaults (authed)", () => {
  test.skip(!SESSION, "set MICROLAB_SESSION_COOKIE (scripts/e2e_session_cookie.py)");

  test.beforeEach(async ({ context }) => {
    await context.addCookies([
      {
        name: "session",
        value: SESSION!,
        domain: "127.0.0.1",
        path: "/",
        httpOnly: true,
        secure: false,
        sameSite: "Lax"
      }
    ]);
  });

  test("selecting a run pre-fills its per-model decoding sliders", async ({ page }, testInfo) => {
    await page.goto("/");
    await page.getByRole("button", { name: /Playground/i }).click();

    const picker = page.getByLabel("Run to serve");
    await expect(picker).toBeEnabled(); // runs loaded from /api/serve/runs

    // A chat run inherits chat defaults: repetition penalty 1.20, top-p 0.90.
    await picker.selectOption("350m-sft-mix");
    await expect(page.getByText(/Repetition penalty 1\.20/)).toBeVisible();
    await expect(page.getByText(/Top-p 0\.90/)).toBeVisible();

    // A base/completion run inherits base defaults: repetition penalty 1.10, top-p 0.95.
    await picker.selectOption("350m");
    await expect(page.getByText(/Repetition penalty 1\.10/)).toBeVisible();
    await expect(page.getByText(/Top-p 0\.95/)).toBeVisible();

    await page.screenshot({
      path: testInfo.outputPath("playground-decoding.png"),
      fullPage: true
    });
  });
});

test.describe("Playground multi-turn chat (authed)", () => {
  test.skip(!SESSION, "set MICROLAB_SESSION_COOKIE (scripts/e2e_session_cookie.py)");

  test.beforeEach(async ({ context }) => {
    await context.addCookies([
      {
        name: "session",
        value: SESSION!,
        domain: "127.0.0.1",
        path: "/",
        httpOnly: true,
        secure: false,
        sameSite: "Lax"
      }
    ]);
  });

  test("a follow-up message carries the prior exchange as history", async ({ page }) => {
    // Mock the serving API: this test pins the FRONTEND contract (what the request body
    // carries across turns) without loading a model onto the GPU mid-training.
    const generateBodies: Array<Record<string, unknown>> = [];
    await page.route("**/api/serve/runs", (route) =>
      route.fulfill({
        json: {
          runs: [
            {
              name: "chat-e2e",
              latest_step: 100,
              mode: "chat",
              decoding: { temperature: 0.8, top_p: 0.9, top_k: 0, repetition_penalty: 1.2 }
            }
          ],
          active: null
        }
      })
    );
    await page.route("**/api/generate", (route) => {
      const body = route.request().postDataJSON() as Record<string, unknown>;
      generateBodies.push(body);
      return route.fulfill({
        status: 200,
        contentType: "text/plain",
        headers: {
          "X-Chat-Turns-Used": String((body.history as unknown[]).length),
          "X-Chat-Turns-Dropped": "0"
        },
        body: `reply ${generateBodies.length}`
      });
    });

    await page.goto("/");
    await page.getByRole("button", { name: /Playground/i }).click();
    await expect(page.getByLabel("Run to serve")).toBeEnabled();

    // Turn 1: send, receive a reply into the transcript, request carried empty history.
    await page.getByLabel("Message").fill("hello model");
    await page.getByRole("button", { name: /^Send$/ }).click();
    await expect(page.getByText("reply 1")).toBeVisible();
    expect(generateBodies[0]).toMatchObject({ prompt: "hello model", history: [] });

    // Turn 2: the follow-up request body carries the completed exchange as history.
    await page.getByLabel("Message").fill("tell me more");
    await page.getByRole("button", { name: /^Send$/ }).click();
    await expect(page.getByText("reply 2")).toBeVisible();
    expect(generateBodies[1]).toMatchObject({
      prompt: "tell me more",
      history: [{ user: "hello model", assistant: "reply 1" }]
    });

    // Both exchanges remain visible and the clear affordance is live.
    await expect(page.getByText("hello model")).toBeVisible();
    await expect(page.getByRole("button", { name: /clear conversation/i })).toBeEnabled();
  });
});
