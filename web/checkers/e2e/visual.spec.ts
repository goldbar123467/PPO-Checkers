import { expect, test } from "@playwright/test";

import { openReady, startGame } from "./helpers";

test("stable desktop landing page", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openReady(page);
  await expect(page).toHaveScreenshot("desktop-landing.png", { fullPage: true });
});

test("stable mobile playing state", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await openReady(page);
  await startGame(page, "Black");
  await expect(page.locator(".move-list li")).toHaveCount(1);
  await expect(page).toHaveScreenshot("mobile-playing.png", {
    fullPage: true,
    // Inference timing varies from run to run.
    mask: [page.locator(".insight__facts")],
  });
});
