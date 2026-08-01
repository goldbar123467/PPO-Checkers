import { expect, test } from "@playwright/test";

test("touch selection makes one legal model-backed move", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start game" }).tap();
  await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
  await page.getByRole("button", { name: /orange man, movable/i }).first().tap();
  await page.getByRole("button", { name: /legal destination/i }).first().tap();
  await expect(page.locator(".move-list li")).toHaveCount(2);
});
