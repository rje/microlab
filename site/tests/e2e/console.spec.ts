import { expect, test } from "@playwright/test";

test("dashboard renders without horizontal overflow", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Phase 0: Evaluation Harness" })).toBeVisible();
  await expect(page.getByText("Measuring Massive Multitask Language Understanding")).toBeVisible();
  await expect(page.getByText("MMLU is a broad multiple-choice benchmark")).toBeVisible();
  await expect(page.getByText("Some evals are best treated as measurement systems")).toBeVisible();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: testInfo.outputPath("dashboard.png"), fullPage: true });
});

test("markdown links open as rendered documents", async ({ page }, testInfo) => {
  await page.goto("/");
  await page.getByRole("link", { name: /environment-setup\.md/i }).click();

  await expect(page.getByRole("dialog", { name: "Markdown document" })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Microlab Python Environment" })).toBeVisible();
  await expect(page.getByText("conda env update -n microlab")).toBeVisible();
  await expect(page.getByRole("link", { name: /raw markdown/i })).toHaveAttribute(
    "href",
    "/plans/environment-setup.md"
  );

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: testInfo.outputPath("markdown-document.png"), fullPage: true });
});
