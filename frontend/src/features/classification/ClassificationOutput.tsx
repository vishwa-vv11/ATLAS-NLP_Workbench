import { ArrowRight, Check, Info, X } from "lucide-react";
import type { ClassificationResult } from "../../api";
import {
  LearningSteps,
  type LearningStep,
} from "../../components/ui/LearningSteps";
import { Matrix } from "../../components/ui/Matrix";
import {
  EmptyState,
  Metric,
  NoteBox,
  SectionLabel,
} from "../../components/ui/Panel";

const CLASSIFICATION_STEPS: LearningStep[] = [
  {
    title: "Turn examples into labeled pairs",
    description:
      "Each CSV row supplies text and the label the classifier should learn to predict.",
  },
  {
    title: "Hold out test examples",
    description:
      "An 80/20 split keeps a portion aside for evaluation. The random seed makes this split reproducible; stratification is used when class counts allow.",
  },
  {
    title: "Fit features and classifier on training data",
    description:
      "The text vectorizer learns its vocabulary from training examples only, then the selected Naive Bayes or Logistic Regression model learns label patterns.",
  },
  {
    title: "Predict unseen labels",
    description:
      "The trained pipeline predicts labels for held-out test texts it did not use during fitting.",
  },
  {
    title: "Review metrics and mistakes",
    description:
      "Accuracy is the share correct; precision measures how often predicted labels are right; recall measures how many true examples are found; weighted F1 balances precision and recall across classes. The confusion matrix and error list show where predictions differ.",
  },
];

export function ClassificationOutput({
  result,
}: {
  result: ClassificationResult | null;
}) {
  if (!result) {
    return (
      <div className="result-body">
        <EmptyState
          title="Ready for a labeled dataset"
          hint="Upload a CSV with text and label columns to train and evaluate."
        />
        <LearningSteps
          title="What this feature does"
          steps={CLASSIFICATION_STEPS}
        />
      </div>
    );
  }

  const { metrics } = result;
  const scores = [
    ["Precision", metrics.precision_weighted],
    ["Recall", metrics.recall_weighted],
    ["F1 score", metrics.f1_weighted],
  ] as const;

  return (
    <div className="result-body">
      <div className="metric-row">
        <Metric
          label="ACCURACY"
          value={`${(metrics.accuracy * 100).toFixed(1)}%`}
          accent
        />
        <Metric label="WEIGHTED F1" value={metrics.f1_weighted.toFixed(3)} />
        <Metric label="TEST SAMPLES" value={result.test_texts.length} />
      </div>
      <section className="result-section">
        <SectionLabel trailing="WEIGHTED AVERAGE">
          CLASSIFICATION METRICS
        </SectionLabel>
        <div className="score-bars">
          {scores.map(([name, score]) => (
            <div className="score-row" key={name}>
              <span>{name}</span>
              <div className="score-track">
                <i style={{ width: `${score * 100}%` }} />
              </div>
              <b>{score.toFixed(3)}</b>
            </div>
          ))}
        </div>
        <p className="result-explanation">
          Precision reflects how many predicted labels were correct; recall
          reflects how many true examples were found. Weighted F1 balances both
          while accounting for class sizes.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing="ACTUAL ROWS · PREDICTED COLUMNS">
          CONFUSION MATRIX
        </SectionLabel>
        <Matrix
          table={metrics.confusion_matrix}
          columns={metrics.labels}
        />
        <p className="result-explanation">
          Rows are actual labels and columns are predictions. Counts on the
          diagonal are correct; off-diagonal counts show which classes were
          confused.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing={`${Math.min(result.test_texts.length, 20)} OF ${result.test_texts.length}`}>
          HELD-OUT PREDICTIONS
        </SectionLabel>
        <div className="prediction-table-wrap">
          <table className="prediction-table">
            <thead>
              <tr>
                <th>TEXT</th>
                <th>ACTUAL</th>
                <th>PREDICTED</th>
                <th>RESULT</th>
              </tr>
            </thead>
            <tbody>
              {result.test_texts.slice(0, 20).map((text, index) => {
                const correct =
                  result.true_labels[index] === result.predictions[index];
                return (
                  <tr key={`${index}-${text}`}>
                    <td title={text}>{text}</td>
                    <td>{result.true_labels[index]}</td>
                    <td>{result.predictions[index]}</td>
                    <td className={correct ? "prediction-correct" : "prediction-wrong"}>
                      {correct ? "Correct" : "Mismatch"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {result.test_texts.length > 20 && (
          <p className="micro-note">
            Showing 20 of {result.test_texts.length} held-out predictions. The
            error list below highlights mistakes.
          </p>
        )}
        <p className="result-explanation">
          Each row compares the model’s predicted label with the known label
          withheld for testing. This is the direct evidence behind the summary
          metrics.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing={`${result.errors.length} ERRORS`}>
          MISCLASSIFIED EXAMPLES
        </SectionLabel>
        {result.errors.length ? (
          <div className="error-list">
            {result.errors.slice(0, 4).map((item, index) => (
              <div
                className="error-row"
                key={`${index}-${item.text}`}
              >
                <span className="error-icon">
                  <X size={13} />
                </span>
                <span className="error-text">{item.text}</span>
                <span className="label-change">
                  {item.true_label} <ArrowRight size={12} />{" "}
                  {item.predicted_label}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-errors">
            <Check size={14} />
            No errors in this split. Small test sets can make this metric
            unstable.
          </div>
        )}
        {result.errors.length > 0 && (
          <p className="result-explanation">
            Each example shows the correct label and the label predicted by the
            model. These cases can reveal ambiguous wording or data patterns
            the model missed.
          </p>
        )}
      </section>
      <NoteBox>
        <Info size={15} />
        <span>
          Metrics are specific to this dataset and random split. Do not infer
          general performance from a small sample.
        </span>
      </NoteBox>
    </div>
  );
}
