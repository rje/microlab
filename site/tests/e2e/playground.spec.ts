import { expect, test } from "@playwright/test";

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
