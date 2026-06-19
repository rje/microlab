import { expect, test } from "@playwright/test";

test("dashboard renders without horizontal overflow", async ({ page }, testInfo) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Phase 0: Evaluation Harness" })).toBeVisible();
  await expect(page.getByText("Measuring Massive Multitask Language Understanding")).toBeVisible();

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  expect(overflow).toBe(false);
  await page.screenshot({ path: testInfo.outputPath("dashboard.png"), fullPage: true });
});
