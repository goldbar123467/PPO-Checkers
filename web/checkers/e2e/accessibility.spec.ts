import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { beginMatch, installMockApi } from "./fixtures";

test("gameplay has no automated WCAG A/AA violations", async ({ page }) => {
  await installMockApi(page);
  await beginMatch(page);
  await page.getByRole("button", { name: /square 9, red man, movable/i }).click();

  const results = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
  expect(results.violations).toEqual([]);
  await expect(page.getByRole("group", { name: /checkers board from red's side/i })).toHaveAttribute(
    "aria-describedby",
    /board-instructions/,
  );
  await expect(page.getByRole("button", { name: /square 9, red man, selected/i })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.getByRole("status")).toContainText("Your move");
});

test("illegal input is rejected with live accessible feedback", async ({ page }) => {
  const api = await installMockApi(page);
  await beginMatch(page);
  await page.getByRole("button", { name: /square 21, white man/i }).click();
  await expect(page.locator("#board-feedback")).toHaveText("Square 21 cannot move in this position.");
  expect(api.moveRequests()).toEqual([]);
});
