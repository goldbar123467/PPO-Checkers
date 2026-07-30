import { expect, test } from "@playwright/test";
import { beginMatch, installMockApi } from "./fixtures";

test("desktop gameplay remains functional without overflow", async ({ page, browserName }) => {
  const consoleProblems: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "warning" || message.type() === "error") consoleProblems.push(message.text());
  });
  page.on("pageerror", (error) => consoleProblems.push(error.message));
  const api = await installMockApi(page);
  await beginMatch(page);
  await page.getByRole("button", { name: /square 9, red man, movable/i }).click();
  await page.getByRole("button", { name: /square 13, empty, legal destination/i }).click();

  await expect(page.getByText(/position · ply 2/i)).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
  expect(api.moveRequests()).toHaveLength(1);
  expect(consoleProblems, `${browserName} console`).toEqual([]);
});
