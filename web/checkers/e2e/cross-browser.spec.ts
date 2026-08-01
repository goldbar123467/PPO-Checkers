import { expect, test } from "@playwright/test";

test("the branded game loads and plays without console errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (message) => { if (message.type() === "error") errors.push(message.text()); });
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Can you beat our checkers AI?" })).toBeVisible();
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: /orange's side/i })).toBeVisible();
  expect(errors).toEqual([]);
});
