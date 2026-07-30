import { expect, test } from "@playwright/test";
import { beginMatch, installMockApi } from "./fixtures";

const viewports = [
  { name: "small phone portrait", width: 320, height: 568 },
  { name: "compact phone portrait", width: 360, height: 640 },
  { name: "modern phone portrait", width: 390, height: 844 },
  { name: "large phone portrait", width: 430, height: 932 },
  { name: "small phone landscape", width: 568, height: 320 },
  { name: "modern phone landscape", width: 844, height: 390 },
  { name: "foldable portrait", width: 540, height: 720 },
  { name: "foldable landscape", width: 720, height: 540 },
  { name: "tablet portrait", width: 768, height: 1024 },
  { name: "tablet landscape", width: 1024, height: 768 },
  { name: "small laptop", width: 1280, height: 720 },
  { name: "desktop", width: 1440, height: 900 },
  { name: "large desktop", width: 1920, height: 1080 },
] as const;

for (const viewport of viewports) {
  test(`${viewport.name} ${viewport.width}x${viewport.height} remains playable`, async ({ page }) => {
    const consoleProblems: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "warning" || message.type() === "error") {
        consoleProblems.push(`${message.type()}: ${message.text()}`);
      }
    });
    page.on("pageerror", (error) => consoleProblems.push(`pageerror: ${error.message}`));

    const api = await installMockApi(page);
    await page.setViewportSize({ width: viewport.width, height: viewport.height });
    await beginMatch(page);

    const metrics = await page.evaluate(() => {
      const root = document.documentElement;
      const board = document.querySelector<HTMLElement>(".board");
      if (!board) throw new Error("Board did not render");
      const boardRect = board.getBoundingClientRect();
      const cells = Array.from(board.querySelectorAll<HTMLElement>(".cell"));
      const firstCellRect = cells[0]?.getBoundingClientRect();
      const playable = Array.from(board.querySelectorAll<HTMLButtonElement>("button.cell"));
      const controlSelectors = [
        "a.brand",
        ".segment",
        "#policy-mode",
        ".primary-action",
        ".text-action",
      ];
      const controls = controlSelectors.flatMap((selector) =>
        Array.from(document.querySelectorAll<HTMLElement>(selector)),
      );
      const panels = [".table-stage", ".game-status", ".setup-panel", ".ledger"]
        .map((selector) => document.querySelector<HTMLElement>(selector)?.getBoundingClientRect())
        .filter((rect): rect is DOMRect => Boolean(rect));
      const overlaps: Array<[number, number]> = [];
      panels.forEach((first, firstIndex) => {
        panels.slice(firstIndex + 1).forEach((second, offset) => {
          const overlapWidth = Math.min(first.right, second.right) - Math.max(first.left, second.left);
          const overlapHeight = Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top);
          if (overlapWidth > 1 && overlapHeight > 1) overlaps.push([firstIndex, firstIndex + offset + 1]);
        });
      });
      const textOverflow = Array.from(
        document.querySelectorAll<HTMLElement>(".panel h2, .panel h3, .primary-action span, .turn-card p"),
      ).filter(
        (element) =>
          element.clientWidth > 1 &&
          element.clientHeight > 1 &&
          element.scrollWidth > element.clientWidth + 1,
      );
      return {
        horizontalOverflow: root.scrollWidth > root.clientWidth,
        board: {
          left: boardRect.left,
          right: boardRect.right,
          top: boardRect.top,
          bottom: boardRect.bottom,
          width: boardRect.width,
          height: boardRect.height,
          documentBottom: root.scrollHeight,
        },
        cellCount: cells.length,
        playableCount: playable.length,
        unequalCells: cells.filter((cell) => {
          const rect = cell.getBoundingClientRect();
          return !firstCellRect || Math.abs(rect.width - firstCellRect.width) > 0.75 ||
            Math.abs(rect.height - firstCellRect.height) > 0.75 ||
            Math.abs(rect.width - rect.height) > 0.75;
        }).length,
        unreachablePlayable: playable.filter((cell) => {
          const rect = cell.getBoundingClientRect();
          return rect.left < boardRect.left - 0.5 || rect.right > boardRect.right + 0.5 ||
            rect.top < boardRect.top - 0.5 || rect.bottom > boardRect.bottom + 0.5 ||
            getComputedStyle(cell).pointerEvents === "none";
        }).length,
        smallControls: controls
          .map((control) => control.getBoundingClientRect())
          .filter((rect) => rect.width > 0 && (rect.width < 44 || rect.height < 44)).length,
        undersizedControlText: controls.filter(
          (control) =>
            control.getBoundingClientRect().width > 0 &&
            Number.parseFloat(getComputedStyle(control).fontSize) < 16,
        ).length,
        overlaps,
        textOverflow: textOverflow.length,
        startButtonBottom: document.querySelector(".primary-action")?.getBoundingClientRect().bottom ?? 0,
      };
    });

    expect(metrics.horizontalOverflow).toBe(false);
    expect(Math.abs(metrics.board.width - metrics.board.height)).toBeLessThanOrEqual(1);
    expect(metrics.board.left).toBeGreaterThanOrEqual(-0.5);
    expect(metrics.board.right).toBeLessThanOrEqual(viewport.width + 0.5);
    expect(metrics.board.top).toBeGreaterThanOrEqual(-0.5);
    expect(metrics.board.bottom).toBeLessThanOrEqual(metrics.board.documentBottom + 0.5);
    expect(metrics.cellCount).toBe(64);
    expect(metrics.playableCount).toBe(32);
    expect(metrics.unequalCells).toBe(0);
    expect(metrics.unreachablePlayable).toBe(0);
    expect(metrics.smallControls).toBe(0);
    expect(metrics.undersizedControlText).toBe(0);
    expect(metrics.overlaps).toEqual([]);
    expect(metrics.textOverflow).toBe(0);
    if (viewport.width > viewport.height && viewport.height < 500) {
      expect(metrics.board.bottom).toBeLessThanOrEqual(viewport.height);
      expect(metrics.startButtonBottom).toBeLessThanOrEqual(viewport.height);
    }

    await page.getByRole("button", { name: /square 9, red man, movable/i }).click();
    await expect(page.getByRole("button", { name: /square 13, empty, legal destination/i })).toBeVisible();
    await page.getByRole("button", { name: /square 13, empty, legal destination/i }).click();
    await expect(page.getByText("21-17")).toBeVisible();
    await expect(page.getByText(/position · ply 2/i)).toBeVisible();
    expect(api.createRequests()).toBe(1);
    expect(api.moveRequests()).toEqual([{ origin: 8, destination: 12 }]);
    expect(consoleProblems).toEqual([]);
  });
}
