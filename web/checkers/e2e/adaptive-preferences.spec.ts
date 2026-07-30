import { expect, test } from "@playwright/test";
import { beginMatch, installMockApi } from "./fixtures";

async function layoutHasNoHorizontalOverflow(page: import("@playwright/test").Page) {
  return page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth);
}

test("200% zoom reflow equivalent and enlarged text retain every essential control", async ({ page }) => {
  await installMockApi(page);
  await page.setViewportSize({ width: 160, height: 284 });
  await beginMatch(page);
  expect(await layoutHasNoHorizontalOverflow(page)).toBe(true);
  await expect(page.locator(".board")).toBeVisible();
  await expect(page.getByRole("button", { name: /start new match/i })).toBeAttached();
  await expect(page.getByRole("status")).toBeAttached();

  await page.setViewportSize({ width: 320, height: 568 });
  await page.addStyleTag({
    content: `
      body { font-size: 125% !important; }
      .segment, select, .primary-action, .turn-card p, .field-help { font-size: 20px !important; }
    `,
  });
  expect(await layoutHasNoHorizontalOverflow(page)).toBe(true);
  await expect(page.getByRole("button", { name: /start new match/i })).toBeAttached();
  await expect(page.getByRole("status")).toBeAttached();
});

test("reduced motion removes nonessential animation without changing move completion", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const api = await installMockApi(page);
  await beginMatch(page);
  const duration = await page.locator(".piece").first().evaluate((element) =>
    Number.parseFloat(getComputedStyle(element).transitionDuration),
  );
  expect(duration).toBeLessThanOrEqual(0.001);

  await page.getByRole("button", { name: /square 9, red man, movable/i }).click();
  await page.getByRole("button", { name: /square 13, empty, legal destination/i }).click();
  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toHaveLength(1);
});

test("2x user page scaling remains enabled and gameplay stays operable", async ({ page }) => {
  const api = await installMockApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  const session = await page.context().newCDPSession(page);

  try {
    await session.send("Emulation.setPageScaleFactor", { pageScaleFactor: 2 });
    await expect
      .poll(() => page.evaluate(() => window.visualViewport?.scale ?? 1))
      .toBeCloseTo(2, 1);
    const viewportPolicy = await page
      .locator('meta[name="viewport"]')
      .getAttribute("content");
    expect(viewportPolicy).not.toMatch(/user-scalable\s*=\s*no/i);
    expect(viewportPolicy).not.toMatch(/maximum-scale\s*=\s*1(?:\D|$)/i);

    await page.getByRole("button", { name: /square 9, red man, movable/i }).focus();
    await page.keyboard.press("Enter");
    await page.getByRole("button", { name: /square 13, empty, legal destination/i }).focus();
    await page.keyboard.press("Enter");
    await expect(page.getByText(/position · ply 2/i)).toBeVisible();
    expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
  } finally {
    await session.send("Emulation.setPageScaleFactor", { pageScaleFactor: 1 });
  }
});

test("board interaction completes under 4x CPU throttling", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);
  const session = await page.context().newCDPSession(page);

  try {
    await session.send("Emulation.setCPUThrottlingRate", { rate: 4 });
    const selectionStarted = Date.now();
    await page.getByRole("button", { name: /square 9, red man, movable/i }).click();
    await expect(page.getByRole("button", { name: /square 13, empty, legal destination/i })).toBeVisible();
    const selectionElapsedMs = Date.now() - selectionStarted;

    const moveStarted = Date.now();
    await page.getByRole("button", { name: /square 13, empty, legal destination/i }).click();
    await expect(page.getByText(/position · ply 2/i)).toBeVisible();
    const moveElapsedMs = Date.now() - moveStarted;

    expect(selectionElapsedMs).toBeLessThan(1_500);
    expect(moveElapsedMs).toBeLessThan(3_000);
    expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
  } finally {
    await session.send("Emulation.setCPUThrottlingRate", { rate: 1 });
  }
});

test("safe-area insets keep the board and controls inside the usable viewport", async ({ page }) => {
  await installMockApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  await page.evaluate(() => {
    const root = document.documentElement.style;
    root.setProperty("--safe-top", "20px");
    root.setProperty("--safe-right", "18px");
    root.setProperty("--safe-bottom", "24px");
    root.setProperty("--safe-left", "22px");
  });

  const metrics = await page.evaluate(() => {
    const shell = document.querySelector<HTMLElement>(".app-shell");
    const board = document.querySelector<HTMLElement>(".board");
    const action = document.querySelector<HTMLElement>(".primary-action");
    if (!shell || !board || !action) throw new Error("Responsive shell did not render");
    const shellStyle = getComputedStyle(shell);
    const boardRect = board.getBoundingClientRect();
    const actionRect = action.getBoundingClientRect();
    return {
      paddingTop: Number.parseFloat(shellStyle.paddingTop),
      paddingRight: Number.parseFloat(shellStyle.paddingRight),
      paddingBottom: Number.parseFloat(shellStyle.paddingBottom),
      paddingLeft: Number.parseFloat(shellStyle.paddingLeft),
      boardLeft: boardRect.left,
      boardRight: boardRect.right,
      actionLeft: actionRect.left,
      actionRight: actionRect.right,
      horizontalOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
    };
  });

  expect(metrics.paddingTop).toBeGreaterThanOrEqual(20);
  expect(metrics.paddingRight).toBeGreaterThanOrEqual(18);
  expect(metrics.paddingBottom).toBeGreaterThanOrEqual(24);
  expect(metrics.paddingLeft).toBeGreaterThanOrEqual(22);
  expect(metrics.boardLeft).toBeGreaterThanOrEqual(22);
  expect(metrics.boardRight).toBeLessThanOrEqual(390 - 18);
  expect(metrics.actionLeft).toBeGreaterThanOrEqual(22);
  expect(metrics.actionRight).toBeLessThanOrEqual(390 - 18);
  expect(metrics.horizontalOverflow).toBe(false);
});
