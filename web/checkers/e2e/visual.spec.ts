import { expect, test } from "@playwright/test";
import {
  beginMatch,
  forcedCapture,
  gameOver,
  installMockApi,
  kingPosition,
} from "./fixtures";

const deviceShots = [
  { name: "small-phone-portrait", width: 320, height: 568 },
  { name: "small-phone-landscape", width: 568, height: 320 },
  { name: "tablet-portrait", width: 768, height: 1024 },
  { name: "tablet-landscape", width: 1024, height: 768 },
  { name: "desktop", width: 1440, height: 900 },
] as const;

for (const shot of deviceShots) {
  test(`stable ${shot.name} gameplay layout`, async ({ page }) => {
    await installMockApi(page);
    await page.setViewportSize({ width: shot.width, height: shot.height });
    await beginMatch(page);
    await page.evaluate(() => scrollTo(0, 0));
    await expect(page).toHaveScreenshot(`${shot.name}.png`, { fullPage: true });
  });
}

test("selected piece and legal destination remain visually distinct", async ({ page }) => {
  await installMockApi(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  await page.getByRole("button", { name: /square 9, red man, movable/i }).click();
  await expect(page.locator(".table-stage")).toHaveScreenshot("selected-and-legal.png");
});

test("forced capture fixture remains visually distinct", async ({ page }) => {
  await installMockApi(page, { initial: forcedCapture });
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  await expect(page.locator(".workspace")).toHaveScreenshot("forced-capture.png");
});

test("king fixture remains visually distinct", async ({ page }) => {
  await installMockApi(page, { initial: kingPosition });
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  await expect(page.locator(".table-stage")).toHaveScreenshot("king-state.png");
});

test("game-over fixture remains visually distinct and locked", async ({ page }) => {
  await installMockApi(page, { initial: gameOver });
  await page.setViewportSize({ width: 390, height: 844 });
  await beginMatch(page);
  await expect(page.getByRole("status")).toContainText("You win");
  await expect(page.locator(".workspace")).toHaveScreenshot("game-over.png");
});
