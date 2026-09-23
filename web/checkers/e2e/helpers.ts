import { expect, type Page } from "@playwright/test";

export async function openReady(page: Page) {
  await page.goto("/");
  await expect(page.getByText("Model ready")).toBeVisible({ timeout: 30_000 });
}

export async function startGame(page: Page, side: "Red" | "Black" = "Red") {
  if (side === "Black") await page.getByRole("button", { name: /Black AI moves first/i }).click();
  await page.getByRole("button", { name: "Start game" }).click();
  await expect(page.getByRole("group", { name: new RegExp(`${side.toLowerCase()}'s side`, "i") })).toBeVisible();
}

export function collectErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(error.message));
  return errors;
}
