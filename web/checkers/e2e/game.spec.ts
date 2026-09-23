import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

import { collectErrors, openReady, startGame } from "./helpers";

const WCAG_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"];

test("the page loads the model in the browser under the production CSP", async ({ page }) => {
  const errors = collectErrors(page);
  const response = await page.goto("/");
  expect(response?.headers()["content-security-policy"]).toContain("script-src 'self'");
  await expect(page).toHaveTitle(/PPO Checkers/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText("trained by self-play");
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("button", { name: "Start game" })).toBeEnabled();
  await expect(page.getByText("354–70–8")).toBeVisible();
  await expect(page.getByText(/not a human skill rating/i)).toBeVisible();
  expect(errors).toEqual([]);
});

test("a player can start a game, move, and get the network's reply", async ({ page }) => {
  await openReady(page);
  await startGame(page);
  await page.locator(".board-move-list summary").click();
  await page.locator(".board-move-list button").first().click();
  await expect(page.locator(".move-list li")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: /Your turn/i })).toBeVisible();
  await expect(page.getByRole("meter", { name: "Value-head position estimate" })).toBeVisible();
  await expect(page.getByText(/ms on device/)).toBeVisible();
});

test("choosing Black lets the network open as Red", async ({ page }) => {
  await openReady(page);
  await startGame(page, "Black");
  await expect(page.locator(".move-list li")).toHaveCount(1);
  await expect(page.getByText(/You are Black/i)).toBeVisible();
  await expect(page.getByRole("heading", { name: /Your turn/i })).toBeVisible();
});

test("the greedy policy answers the same opening the same way", async ({ page }) => {
  const replies: string[] = [];
  for (let game = 0; game < 2; game += 1) {
    await openReady(page);
    await startGame(page);
    await page.getByRole("button", { name: /square 11, red man, movable/i }).click();
    await page.getByRole("button", { name: /square 15, empty, legal destination/i }).click();
    await expect(page.locator(".move-list li")).toHaveCount(2);
    replies.push((await page.locator(".move-list li").nth(1).locator("strong").textContent()) ?? "");
  }
  expect(replies[0]).toMatch(/^\d+-\d+$/);
  expect(replies[1]).toBe(replies[0]);
});

test("the board remains fully keyboard playable", async ({ page }) => {
  await openReady(page);
  await startGame(page);
  const origin = page.getByRole("button", { name: /red man, movable/i }).first();
  await origin.focus();
  await page.keyboard.press("Enter");
  const destination = page.getByRole("button", { name: /legal destination/i }).first();
  await destination.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".move-list li")).toHaveCount(2);
});

test("the page and an active game have no automated WCAG A or AA violations", async ({ page }) => {
  await openReady(page);
  expect((await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze()).violations).toEqual([]);
  await startGame(page);
  await page.locator(".board-move-list summary").click();
  await page.locator(".board-move-list button").first().click();
  await expect(page.locator(".move-list li")).toHaveCount(2);
  expect((await new AxeBuilder({ page }).withTags(WCAG_TAGS).analyze()).violations).toEqual([]);
});

test("skip navigation, reduced motion, and high contrast remain available", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await openReady(page);
  await page.keyboard.press("Tab");
  const skip = page.getByRole("link", { name: "Skip to the checkers game" });
  await expect(skip).toBeFocused();
  await expect(skip).toHaveCSS("opacity", "1");
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior)).toBe("auto");
  await startGame(page);
  await page.getByRole("checkbox", { name: "High-contrast board" }).check();
  await expect(page.locator(".site-shell")).toHaveClass(/board-high-contrast/);
});

for (const viewport of [
  { width: 320, height: 568 },
  { width: 375, height: 667 },
  { width: 390, height: 844 },
  { width: 768, height: 1024 },
  { width: 1024, height: 768 },
  { width: 1366, height: 768 },
  { width: 1920, height: 1080 },
  { width: 2560, height: 1440 },
]) {
  test(`${viewport.width}x${viewport.height} has no horizontal overflow and keeps the game usable`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await openReady(page);
    await startGame(page);
    expect(
      await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1),
    ).toBe(true);
    const width = await page.locator(".board").evaluate((board) => board.getBoundingClientRect().width);
    expect(width).toBeGreaterThanOrEqual(viewport.width === 320 ? 270 : 300);
  });
}

test("a failed weight download is clear and recoverable", async ({ page }) => {
  let failures = 0;
  await page.route("**/assets/policy-*.bin", async (route) => {
    if (failures < 1) {
      failures += 1;
      await route.fulfill({ status: 503, body: "unavailable" });
      return;
    }
    await route.continue();
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("HTTP 503");
  await expect(page.getByRole("button", { name: "Model unavailable" })).toBeDisabled();
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByRole("alert")).toHaveCount(0);
});
