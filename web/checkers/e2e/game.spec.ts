import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";

test("the IMSA game-first page loads with real identity, evidence, and no browser errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  page.on("pageerror", (error) => errors.push(error.message));

  await page.goto("/");
  await expect(page).toHaveTitle("IMSA West Checkers AI");
  await expect(page.getByRole("heading", { name: "Can you beat our checkers AI?" })).toBeVisible();
  const logo = page.getByAltText("Indiana Math and Science Academy West logo");
  await expect(logo).toBeVisible();
  await expect.poll(() => logo.evaluate((image: HTMLImageElement) => image.complete && image.naturalWidth === 447)).toBe(true);
  await expect(page.getByText("354–70–8")).toBeVisible();
  await expect(page.getByText(/not a human skill rating/i)).toBeVisible();
  await expect(page.getByRole("button", { name: "Start game" })).toHaveCount(1);
  expect(errors).toEqual([]);
});

test("a student can start a real game and submit a legal move", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
  await page.locator(".board-move-list summary").click();
  await page.locator(".board-move-list button").first().click();
  await expect(page.locator(".move-list li")).toHaveCount(2);
  await expect(page.getByRole("heading", { name: /Your turn/i })).toBeVisible();
});

test("side selection stays simple and white lets the saved policy open", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /White AI moves first/i }).click();
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: /white's side/i })).toBeVisible();
  await expect(page.locator(".move-list li")).toHaveCount(1);
  await expect(page.getByText(/You are white/i)).toBeVisible();
});

test("the board remains fully keyboard playable", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Start game" }).click();
  const origin = page.getByRole("button", { name: /orange man, movable/i }).first();
  await origin.focus();
  await page.keyboard.press("Enter");
  const destination = page.getByRole("button", { name: /legal destination/i }).first();
  await destination.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator(".move-list li")).toHaveCount(2);
});

test("the page and active game have no automated WCAG A or AA violations", async ({ page }) => {
  await page.goto("/");
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze()).violations).toEqual([]);
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
  expect((await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"]).analyze()).violations).toEqual([]);
});

test("skip navigation, reduced motion, and board contrast remain available", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.keyboard.press("Tab");
  await expect(page.getByRole("link", { name: "Skip to the checkers game" })).toBeFocused();
  await expect(page.getByRole("link", { name: "Skip to the checkers game" })).toHaveCSS("opacity", "1");
  expect(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior)).toBe("auto");
  await page.getByRole("button", { name: "Start game" }).click();
  await page.getByRole("checkbox", { name: "Stronger board contrast" }).check();
  await expect(page.locator(".site-shell")).toHaveClass(/board-high-contrast/);
});

for (const viewport of [
  { width: 320, height: 568 },
  { width: 375, height: 667 },
  { width: 390, height: 844 },
  { width: 430, height: 932 },
  { width: 768, height: 1024 },
  { width: 1024, height: 768 },
  { width: 1366, height: 768 },
  { width: 1440, height: 900 },
  { width: 1920, height: 1080 },
  { width: 2560, height: 1440 },
]) {
  test(`${viewport.width}x${viewport.height} has no horizontal overflow and keeps the game usable`, async ({ page }) => {
    await page.setViewportSize(viewport);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "Start game" })).toBeVisible();
    await page.getByRole("button", { name: "Start game" }).click();
    await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
    expect(await page.locator(".board").evaluate((board) => board.getBoundingClientRect().width)).toBeGreaterThanOrEqual(viewport.width === 320 ? 270 : 300);
  });
}

test("a model-loading failure is clear and recoverable", async ({ page }) => {
  let failures = 0;
  await page.route("**/api/model", async (route) => {
    if (failures < 2) {
      failures += 1;
      await route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: { code: "model_unavailable", message: "Policy loading test failure." } }) });
      return;
    }
    await route.continue();
  });
  await page.goto("/");
  await expect(page.getByRole("alert")).toContainText("Policy loading test failure");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByText(/Policy update 4,608 ready/i)).toBeVisible();
});
