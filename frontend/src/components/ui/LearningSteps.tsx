import type { ReactNode } from "react";

export interface LearningStep {
  title: string;
  output?: ReactNode;
  description: ReactNode;
}

export function LearningSteps({
  title = "How this works",
  steps,
}: {
  title?: string;
  steps: LearningStep[];
}) {
  return (
    <section className="learning-steps" aria-label={title}>
      <div className="learning-steps-heading">
        <span className="learning-heading-mark" aria-hidden="true">
          ↳
        </span>
        <div>
          <strong>{title}</strong>
          <span>The process behind these results</span>
        </div>
      </div>
      <ol>
        {steps.map((step, index) => (
          <li key={step.title}>
            <span className="learning-step-number">
              {String(index + 1).padStart(2, "0")}
            </span>
            <div>
              <strong>{step.title}</strong>
              {step.output !== undefined && (
                <div className="learning-step-output">{step.output}</div>
              )}
              <p>{step.description}</p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
