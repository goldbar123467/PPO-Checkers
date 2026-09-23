import { expect, test } from "@playwright/test";

import { collectErrors, openReady, startGame } from "./helpers";

test("the model loads and plays without console errors", async ({ page }) => {
  const errors = collectErrors(page);
  await openReady(page);
  await startGame(page, "Black");
  await expect(page.locator(".move-list li")).toHaveCount(1);
  expect(errors).toEqual([]);
});
