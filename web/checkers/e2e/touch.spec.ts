import { expect, test, type CDPSession, type Page } from "@playwright/test";
import {
  afterMultiJump,
  beginMatch,
  forcedCapture,
  forcedContinuation,
  installMockApi,
} from "./fixtures";

async function touchPoint(page: Page, selector: string, xRatio = 0.5, yRatio = 0.5) {
  const box = await page.locator(selector).boundingBox();
  if (!box) throw new Error(`No touch box for ${selector}`);
  return { x: box.x + box.width * xRatio, y: box.y + box.height * yRatio };
}

async function dispatchTouch(
  session: CDPSession,
  type: "touchStart" | "touchMove" | "touchEnd" | "touchCancel",
  point?: { x: number; y: number },
) {
  await session.send("Input.dispatchTouchEvent", {
    type,
    touchPoints: point ? [{ x: point.x, y: point.y, radiusX: 1, radiusY: 1 }] : [],
  });
}

async function dispatchPenClick(session: CDPSession, point: { x: number; y: number }) {
  await session.send("Input.dispatchMouseEvent", {
    type: "mousePressed",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 1,
    clickCount: 1,
    pointerType: "pen",
  });
  await session.send("Input.dispatchMouseEvent", {
    type: "mouseReleased",
    x: point.x,
    y: point.y,
    button: "left",
    buttons: 0,
    clickCount: 1,
    pointerType: "pen",
  });
}

test("one tap sequence produces exactly one logical move", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);

  await page.getByRole("button", { name: /square 9, red man, movable/i }).tap();
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toBeVisible();
  await expect(page.getByRole("button", { name: /square 13, empty, legal destination/i })).toBeVisible();
  expect(
    await page.getByRole("button", { name: /clear selection/i }).evaluate((element) =>
      Number.parseFloat(getComputedStyle(element).fontSize),
    ),
  ).toBeGreaterThanOrEqual(16);
  await page.getByRole("button", { name: /square 13, empty, legal destination/i }).tap();

  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
});

test("near-boundary taps map to the visible cell and outside taps are ignored", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);
  const boardBox = await page.locator(".board").boundingBox();
  if (!boardBox) throw new Error("Board missing");

  await page.touchscreen.tap(Math.max(1, boardBox.x - 3), boardBox.y + boardBox.height / 2);
  await expect(page.locator(".is-selected")).toHaveCount(0);

  const nearOriginEdge = await touchPoint(page, '[data-square-index="8"]', 0.02, 0.5);
  await page.touchscreen.tap(nearOriginEdge.x, nearOriginEdge.y);
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toBeVisible();

  const nearDestinationEdge = await touchPoint(page, '[data-square-index="12"]', 0.98, 0.5);
  await page.touchscreen.tap(nearDestinationEdge.x, nearDestinationEdge.y);
  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
});

test("forced capture and multi-jump keep the required piece selected", async ({ page }) => {
  const api = await installMockApi(page, {
    initial: forcedCapture,
    moveResponses: [forcedContinuation, afterMultiJump],
  });
  await beginMatch(page);

  await expect(page.getByRole("button", { name: /square 13, red man, selected/i })).toBeVisible();
  await page.getByRole("button", { name: /square 22, empty, legal destination/i }).tap();
  await expect(page.getByRole("button", { name: /square 22, red man, selected/i })).toBeVisible();
  await page.getByRole("button", { name: /square 31, empty, legal destination/i }).tap();

  await expect(page.getByRole("button", { name: /square 31, red king/i })).toBeDisabled();
  expect(api.moveRequests()).toEqual([
    { origin: 12, destination: 21 },
    { origin: 21, destination: 30 },
  ]);
});

test("rapid repeated destination taps cannot duplicate a request", async ({ page }) => {
  const api = await installMockApi(page, { moveDelayMs: 80 });
  await beginMatch(page);
  await page.getByRole("button", { name: /square 9, red man, movable/i }).tap();
  const destination = page.getByRole("button", { name: /square 13, empty, legal destination/i });

  await Promise.allSettled([destination.tap(), destination.tap(), destination.tap()]);
  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
});

test("pointer cancellation, release outside, light squares, and edge-adjacent taps are rejected", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);
  const session = await page.context().newCDPSession(page);
  const origin = await touchPoint(page, '[data-square-index="8"]');
  const boardBox = await page.locator(".board").boundingBox();
  if (!boardBox) throw new Error("Board missing");

  await dispatchTouch(session, "touchStart", origin);
  await dispatchTouch(session, "touchCancel");
  await expect(page.locator(".is-selected")).toHaveCount(0);

  await dispatchTouch(session, "touchStart", origin);
  await dispatchTouch(session, "touchMove", { x: Math.max(1, boardBox.x - 3), y: origin.y });
  await dispatchTouch(session, "touchEnd");
  await expect(page.locator(".is-selected")).toHaveCount(0);

  await page.touchscreen.tap(
    boardBox.x + boardBox.width / 16,
    boardBox.y + boardBox.height / 16,
  );
  await expect(page.locator(".is-selected")).toHaveCount(0);

  await page.touchscreen.tap(origin.x, origin.y);
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toBeVisible();
  expect(api.moveRequests()).toEqual([]);
});

test("board gestures do not scroll while normal page gestures still do", async ({ page }) => {
  await installMockApi(page);
  await beginMatch(page);
  const session = await page.context().newCDPSession(page);
  await page.evaluate(() => {
    document.documentElement.style.scrollBehavior = "auto";
    scrollTo(0, 0);
  });
  expect(await page.evaluate(() => scrollY)).toBe(0);
  const boardCenter = await touchPoint(page, ".board");

  await dispatchTouch(session, "touchStart", boardCenter);
  await dispatchTouch(session, "touchMove", { x: boardCenter.x, y: boardCenter.y - 35 });
  await dispatchTouch(session, "touchEnd");
  expect(await page.evaluate(() => scrollY)).toBe(0);
  expect(await page.locator(".board").evaluate((element) => getComputedStyle(element).touchAction)).toBe("none");

  const outside = { x: 370, y: 760 };
  await dispatchTouch(session, "touchStart", outside);
  await dispatchTouch(session, "touchMove", { x: outside.x, y: 320 });
  await dispatchTouch(session, "touchEnd");
  await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(0);
  expect(await page.locator(".setup-panel").evaluate((element) => getComputedStyle(element).touchAction)).not.toBe("none");
});

test("pen pointer input uses the same exactly-once move pathway", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);
  const session = await page.context().newCDPSession(page);

  await dispatchPenClick(session, await touchPoint(page, '[data-square-index="8"]'));
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toBeVisible();
  await dispatchPenClick(session, await touchPoint(page, '[data-square-index="12"]'));

  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
});
