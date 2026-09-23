import type { PolicyInsight } from "@/types";

interface ModelInsightProps {
  insight: PolicyInsight | null;
  thinking: boolean;
}

function evaluationLabel(value: number): string {
  if (value > 0.25) return "favors the AI";
  if (value < -0.25) return "favors you";
  return "roughly even";
}

export function ModelInsight({ insight, thinking }: ModelInsightProps) {
  const position = insight ? ((insight.value + 1) / 2) * 100 : 50;
  return (
    <section className="panel insight" aria-labelledby="insight-heading">
      <p className="panel-label" id="insight-heading">What the network saw</p>
      {insight ? (
        <>
          <div className="insight__meter">
            <div className="insight__labels" aria-hidden="true">
              <span>You</span>
              <span>AI</span>
            </div>
            <div
              className="insight__track"
              role="meter"
              aria-label="Value-head position estimate"
              aria-valuemin={-1}
              aria-valuemax={1}
              aria-valuenow={Number(insight.value.toFixed(2))}
              aria-valuetext={`${insight.value.toFixed(2)}, ${evaluationLabel(insight.value)}`}
            >
              <span style={{ left: `${position}%` }} />
            </div>
            <p>
              Evaluation <strong>{insight.value >= 0 ? "+" : ""}{insight.value.toFixed(2)}</strong>{" "}
              {evaluationLabel(insight.value)}
            </p>
          </div>
          <dl className="insight__facts">
            <div>
              <dt>Move confidence</dt>
              <dd>{Math.round(insight.confidence * 100)}%<small> of {insight.legalMoveCount}</small></dd>
            </div>
            <div>
              <dt>Inference</dt>
              <dd>{Math.max(1, Math.round(insight.inferenceMs))} ms<small> on device</small></dd>
            </div>
          </dl>
        </>
      ) : (
        <p className="insight__empty">
          {thinking ? "The AI is evaluating its first move…" : "After the AI moves, its value estimate and move confidence appear here."}
        </p>
      )}
    </section>
  );
}
