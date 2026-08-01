import { expect, test } from "@playwright/test";

test("stable IMSA game-first desktop", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/");
  await expect(page.getByText(/Policy update 4,608 ready/i)).toBeVisible();
  await expect(page.getByText("37.7M")).toBeVisible();
  await expect(page.getByText("50.3M")).toHaveCount(0);
  await expect(page).toHaveScreenshot("imsa-game-desktop.png", { fullPage: true });
});

test("stable IMSA mobile playing state", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
  await expect(page.getByText("37.7M")).toBeVisible();
  await expect(page.getByText("50.3M")).toHaveCount(0);
  await expect(page).toHaveScreenshot("imsa-game-mobile-playing.png", { fullPage: true });
});
