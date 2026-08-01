import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { chromium } from "playwright";

const baseUrl = process.argv[2] ?? "http://127.0.0.1:8765/";
const outputPath = resolve(process.argv[3] ?? "../../reports/checkers_web_performance_20260801.json");
const browser = await chromium.launch({ headless: true, args: ["--js-flags=--expose-gc"] });
const page = await browser.newPage({ viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 });
const cdp = await page.context().newCDPSession(page);

await cdp.send("Network.enable");
await cdp.send("Performance.enable");
await cdp.send("Network.emulateNetworkConditions", {
  offline: false,
  latency: 40,
  downloadThroughput: (10 * 1024 * 1024) / 8,
  uploadThroughput: (5 * 1024 * 1024) / 8,
});
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });

await page.addInitScript(() => {
  globalThis.__checkersVitals = { cls: 0, interactionDurations: [], lcpMs: 0 };
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) globalThis.__checkersVitals.lcpMs = entry.startTime;
  }).observe({ type: "largest-contentful-paint", buffered: true });
  new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      if (!entry.hadRecentInput) globalThis.__checkersVitals.cls += entry.value;
    }
  }).observe({ type: "layout-shift", buffered: true });
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.interactionId) globalThis.__checkersVitals.interactionDurations.push(entry.duration);
      }
    }).observe({ type: "event", buffered: true, durationThreshold: 0 });
  } catch {
    // The report fails its Event Timing gate if the browser does not expose entries.
  }
});

const response = await page.goto(baseUrl, { waitUntil: "networkidle" });
if (!response?.ok()) throw new Error(`Opening page returned ${response?.status() ?? "no response"}.`);
await page.getByRole("heading", { name: "Can you beat our checkers AI?" }).waitFor();
await page.waitForTimeout(250);

const initial = await page.evaluate(() => ({
  resources: performance.getEntriesByType("resource").map((entry) => ({
    bytes: entry.encodedBodySize || entry.transferSize || 0,
    durationMs: Number(entry.duration.toFixed(2)),
    kind: entry.initiatorType,
    url: entry.name,
  })),
  navigationMs: Number((performance.getEntriesByType("navigation")[0]?.duration ?? 0).toFixed(2)),
  vitals: globalThis.__checkersVitals,
}));

const resources = [];
for (const resource of initial.resources) {
  const assetResponse = await page.request.get(resource.url);
  if (!assetResponse.ok()) throw new Error(`Measured resource returned ${assetResponse.status()}: ${resource.url}`);
  const body = await assetResponse.body();
  resources.push({ ...resource, sha256: createHash("sha256").update(body).digest("hex") });
}

const totalInitialBytes = resources.reduce((total, resource) => total + resource.bytes, 0);
const imageBytes = resources.filter((resource) => resource.kind === "img").reduce((total, resource) => total + resource.bytes, 0);

const boardStartedAt = await page.evaluate(() => performance.now());
await page.getByRole("button", { name: "Start game" }).click();
await page.getByRole("group", { name: /orange's side/i }).waitFor();
const boardReadyMs = await page.evaluate((startedAt) => performance.now() - startedAt, boardStartedAt);

await page.locator(".board-move-list summary").click();
await page.evaluate(() => { globalThis.__checkersVitals.interactionDurations = []; });
const moveStartedAt = await page.evaluate(() => performance.now());
await page.locator(".board-move-list button").first().click();
await page.waitForFunction(() => document.querySelectorAll(".move-list li").length >= 2);
const policyRoundTripMs = await page.evaluate((startedAt) => performance.now() - startedAt, moveStartedAt);
await page.waitForTimeout(100);

const interactionEventLatencyMs = await page.evaluate(() => {
  const durations = globalThis.__checkersVitals.interactionDurations;
  return durations.length ? Math.max(...durations) : null;
});

async function readHeapBytes() {
  await cdp.send("HeapProfiler.collectGarbage");
  const metrics = await cdp.send("Performance.getMetrics");
  return metrics.metrics.find((metric) => metric.name === "JSHeapUsedSize")?.value ?? null;
}

const firstHeap = await readHeapBytes();
const heapSamples = [{ game: 1, usedBytes: firstHeap }];
for (let gameNumber = 2; gameNumber <= 6; gameNumber += 1) {
  const createResponse = page.waitForResponse((candidate) => candidate.url().includes("/api/games") && candidate.request().method() === "POST");
  await page.getByRole("button", { name: "Start a new game" }).click();
  if (!(await createResponse).ok()) throw new Error(`Game ${gameNumber} failed to start.`);
  await page.getByRole("group", { name: /orange's side/i }).waitFor();
  heapSamples.push({ game: gameNumber, usedBytes: await readHeapBytes() });
}

const heapValues = heapSamples.map((sample) => sample.usedBytes).filter((value) => value !== null);
const heapGrowthBytes = heapValues.length > 1 ? Math.max(...heapValues) - heapValues[0] : null;
const finalVitals = await page.evaluate(() => globalThis.__checkersVitals);

const report = {
  schemaVersion: 4,
  measuredAt: new Date().toISOString(),
  context: {
    browser: await browser.version(),
    cpu: "4x Chromium CPU slowdown",
    network: "local loopback with 40 ms latency, 10 Mbps down, 5 Mbps up",
    url: baseUrl,
    viewport: { width: 1366, height: 768 },
  },
  budgets: {
    boardReadyMsMaximum: 1000,
    clsMaximum: 0.05,
    imageBytesMaximum: 64 * 1024,
    interactionEventLatencyMsMaximum: 200,
    lcpMsMaximum: 2500,
    policyRoundTripMsMaximum: 1000,
    retainedHeapGrowthBytesMaximum: 10 * 1024 * 1024,
    totalInitialBytesMaximum: 375 * 1024,
  },
  measurements: {
    boardReadyMs: Number(boardReadyMs.toFixed(2)),
    cls: Number(finalVitals.cls.toFixed(4)),
    heapGrowthBytes,
    heapSamples,
    imageBytes,
    interactionEventLatencyMs: interactionEventLatencyMs === null ? null : Number(interactionEventLatencyMs.toFixed(2)),
    lcpMs: Number(initial.vitals.lcpMs.toFixed(2)),
    navigationMs: initial.navigationMs,
    policyRoundTripMs: Number(policyRoundTripMs.toFixed(2)),
    totalInitialBytes,
  },
  resources: resources.sort((left, right) => right.bytes - left.bytes),
};

report.passed =
  report.measurements.boardReadyMs <= report.budgets.boardReadyMsMaximum &&
  report.measurements.cls <= report.budgets.clsMaximum &&
  report.measurements.imageBytes <= report.budgets.imageBytesMaximum &&
  report.measurements.interactionEventLatencyMs !== null &&
  report.measurements.interactionEventLatencyMs <= report.budgets.interactionEventLatencyMsMaximum &&
  report.measurements.lcpMs <= report.budgets.lcpMsMaximum &&
  report.measurements.policyRoundTripMs <= report.budgets.policyRoundTripMsMaximum &&
  (report.measurements.heapGrowthBytes === null || report.measurements.heapGrowthBytes <= report.budgets.retainedHeapGrowthBytesMaximum) &&
  report.measurements.totalInitialBytes <= report.budgets.totalInitialBytesMaximum;

await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
await browser.close();

console.log(JSON.stringify({ outputPath, passed: report.passed, measurements: report.measurements }));
if (!report.passed) process.exitCode = 1;
