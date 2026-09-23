import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import App from "@/App";
import type { PolicyRunner } from "@/lib/policyRunner";

function fakeRunner(overrides: Partial<PolicyRunner> = {}): PolicyRunner {
  return {
    evaluate: vi.fn(async () => ({ logits: new Float32Array(128), value: 0.4, inferenceMs: 3.2 })),
    dispose: vi.fn(),
    ...overrides,
  };
}

function renderApp(createRunner: () => Promise<PolicyRunner>) {
  return render(<App moveDelayMs={0} createRunner={createRunner} />);
}

async function playFirstLegalMove() {
  fireEvent.click(screen.getByText("Legal move list"));
  const list = screen.getByRole("list", { name: "Legal move list" });
  fireEvent.click(within(list).getAllByRole("button")[0]);
}

describe("PPO Checkers application", () => {
  it("presents the project, its evidence, and a ready game setup", async () => {
    const createRunner = vi.fn(async () => fakeRunner());
    renderApp(createRunner);

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(
      "Play checkers against a neural network trained by self-play.",
    );
    expect(screen.getByRole("button", { name: "Loading model…" })).toBeDisabled();
    expect(await screen.findByRole("button", { name: "Start game" })).toBeEnabled();
    expect(screen.getByText("Model ready")).toBeInTheDocument();
    expect(screen.getByText("354–70–8")).toBeInTheDocument();
    expect(screen.getByText(/not a human skill rating/i)).toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "GitHub" })[0]).toHaveAttribute(
      "href",
      "https://github.com/goldbar123467/PPO-Checkers",
    );
    expect(createRunner).toHaveBeenCalled();
  });

  it("lets the AI open as Red when the player picks Black, then shows its evaluation", async () => {
    const runner = fakeRunner();
    renderApp(async () => runner);

    fireEvent.click(await screen.findByRole("button", { name: /Black AI moves first/i }));
    fireEvent.click(screen.getByRole("button", { name: "Start game" }));

    expect(await screen.findByRole("heading", { name: /Your turn/i })).toBeInTheDocument();
    expect(screen.getByRole("group", { name: /black's side/i })).toBeInTheDocument();
    expect(screen.getByText("1 move")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: "Value-head position estimate" })).toHaveAttribute(
      "aria-valuenow",
      "0.4",
    );
    expect(screen.getByText(/favors the AI/)).toBeInTheDocument();
    expect(screen.getByText("3 ms")).toBeInTheDocument();
    expect(runner.evaluate).toHaveBeenCalledOnce();
  });

  it("applies a human move and the AI's reply", async () => {
    const runner = fakeRunner();
    renderApp(async () => runner);

    fireEvent.click(await screen.findByRole("button", { name: "Start game" }));
    expect(await screen.findByRole("group", { name: /red's side/i })).toBeInTheDocument();
    expect(runner.evaluate).not.toHaveBeenCalled();
    await playFirstLegalMove();

    expect(await screen.findByText("2 moves")).toBeInTheDocument();
    expect(await screen.findByRole("heading", { name: /Your turn/i })).toBeInTheDocument();
    expect(runner.evaluate).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Start a new game" }));
    expect(await screen.findByText("0 moves")).toBeInTheDocument();
  });

  it("recovers from a failed model download", async () => {
    const createRunner = vi
      .fn<() => Promise<PolicyRunner>>()
      .mockRejectedValueOnce(new Error("Weights test failure."))
      .mockResolvedValue(fakeRunner());
    renderApp(createRunner);

    expect(await screen.findByRole("alert")).toHaveTextContent("Weights test failure.");
    expect(screen.getByText("Model failed to load")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByRole("button", { name: "Start game" })).toBeEnabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("reports an interrupted AI turn and resumes it on retry", async () => {
    const evaluate = vi
      .fn<PolicyRunner["evaluate"]>()
      .mockRejectedValueOnce(new Error("Inference test failure."))
      .mockResolvedValue({ logits: new Float32Array(128), value: -0.6, inferenceMs: 0.2 });
    renderApp(async () => fakeRunner({ evaluate }));

    fireEvent.click(await screen.findByRole("button", { name: /Black AI moves first/i }));
    fireEvent.click(screen.getByRole("button", { name: "Start game" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Inference test failure.");

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());
    expect(await screen.findByText(/favors you/)).toBeInTheDocument();
    expect(screen.getByText("1 ms")).toBeInTheDocument();
  });

  it("toggles the high-contrast board", async () => {
    renderApp(async () => fakeRunner());
    const toggle = screen.getByRole("checkbox", { name: "High-contrast board" });
    fireEvent.click(toggle);
    expect(document.querySelector(".site-shell")).toHaveClass("board-high-contrast");
    await screen.findByText("Model ready");
  });
});
