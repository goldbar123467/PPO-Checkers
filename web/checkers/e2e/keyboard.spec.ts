import { expect, test } from "@playwright/test";
import { beginMatch, installMockApi } from "./fixtures";

test("tab order, visible focus, board navigation, selection, Escape, and confirmation work", async ({ page }) => {
  const api = await installMockApi(page);
  await page.goto("/");

  await page.keyboard.press("Tab");
  const homeLink = page.getByRole("link", { name: /checkers home/i });
  await expect(homeLink).toBeFocused();
  expect(
    Number.parseFloat(await homeLink.evaluate((element) => getComputedStyle(element).outlineWidth)),
  ).toBeGreaterThanOrEqual(3);
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "Red" })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "White" })).toBeFocused();
  await page.keyboard.press("Space");
  await expect(page.getByRole("button", { name: "White" })).toHaveAttribute("aria-pressed", "true");
  await page.getByRole("button", { name: "Red" }).click();

  await page.getByRole("button", { name: "Begin match" }).click();
  await page.getByRole("group", { name: /checkers board/i }).waitFor();
  const origin = page.getByRole("button", { name: /square 9, red man, movable/i });
  await origin.focus();
  const outline = await origin.evaluate((element) => getComputedStyle(element).outlineWidth);
  expect(Number.parseFloat(outline)).toBeGreaterThanOrEqual(3);

  await page.keyboard.press("Enter");
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("button", { name: /clear selection/i })).toHaveCount(0);

  await page.keyboard.press("Enter");
  await page.keyboard.press("ArrowUp");
  await expect(page.locator('[data-square-index="13"]')).toBeFocused();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator('[data-square-index="12"]')).toBeFocused();
  await page.keyboard.press("Enter");

  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
});

test("the board exposes only one tab stop and cannot trap keyboard focus", async ({ page }) => {
  await installMockApi(page);
  await beginMatch(page);
  await expect(page.locator('.board button[tabindex="0"]')).toHaveCount(1);
  await expect(page.locator('.board button[tabindex="-1"]')).toHaveCount(31);

  await page.locator('.board button[tabindex="0"]').focus();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("status")).not.toBeFocused();
  expect(await page.evaluate(() => document.activeElement?.closest(".board") === null)).toBe(true);
});
