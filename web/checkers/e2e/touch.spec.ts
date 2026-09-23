import { expect, test } from "@playwright/test";

import { openReady } from "./helpers";

test("touch selection makes one legal move and gets one reply", async ({ page }) => {
  await openReady(page);
  await page.getByRole("button", { name: "Start game" }).tap();
  await expect(page.getByRole("group", { name: /red's side/i })).toBeVisible();
  await page.getByRole("button", { name: /red man, movable/i }).first().tap();
  await page.getByRole("button", { name: /legal destination/i }).first().tap();
  await expect(page.locator(".move-list li")).toHaveCount(2);
});
