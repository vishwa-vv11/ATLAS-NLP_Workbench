import { Info } from "lucide-react";
import type { LanguageModelResult } from "../../api";
import {
  LearningSteps,
  type LearningStep,
} from "../../components/ui/LearningSteps";
import {
  EmptyState,
  Metric,
  NoteBox,
  SectionLabel,
} from "../../components/ui/Panel";

const LANGUAGE_STEPS: LearningStep[] = [
  {
    title: "Prepare sentences",
    description:
      "Training and evaluation text are split into sentences and word tokens separately.",
  },
  {
    title: "Count adjacent word pairs",
    description:
      "The model records bigrams, including sentence-start and sentence-end markers, to estimate which word tends to follow another.",
  },
  {
    title: "Smooth sparse and unseen pairs",
    description:
      "Add-one (Laplace) smoothing gives every possible next token a non-zero probability. Unseen evaluation words map to <UNK>.",
  },
  {
    title: "Score held-out text",
    description:
      "For each evaluation token, the model estimates its probability given the preceding token, then combines those probabilities into perplexity.",
  },
  {
    title: "Interpret the score carefully",
    description:
      "Lower perplexity means this model found the evaluation sequence less surprising. Compare scores only with consistent tokenization and genuinely held-out text.",
  },
];

export function LanguageOutput({
  result,
}: {
  result: LanguageModelResult | null;
}) {
  if (!result) {
    return (
      <div className="result-body">
        <EmptyState
          title="Score held-out text"
          hint="Provide separate training and evaluation passages to calculate perplexity."
        />
        <LearningSteps
          title="What this feature does"
          steps={LANGUAGE_STEPS}
        />
      </div>
    );
  }

  return (
    <div className="result-body">
      <div className="perplexity-display">
        <div className="perplexity-orbit">
          <span>PP</span>
          <strong>{result.perplexity.toFixed(2)}</strong>
        </div>
        <div>
          <SectionLabel>HELD-OUT PERPLEXITY</SectionLabel>
          <p>
            Lower values indicate the model assigns higher probability to this
            evaluation text.
          </p>
        </div>
      </div>
      <div className="metric-row">
        <Metric label="TRAIN SENTENCES" value={result.training_sentences} />
        <Metric label="EVAL SENTENCES" value={result.evaluation_sentences} />
        <Metric label="SMOOTHING" value="α = 1" accent />
      </div>
      <p className="result-explanation">
        Sentence counts show how much text was used for fitting and scoring.
        α = 1 adds one pseudo-count to each possible bigram so unseen pairs do
        not receive zero probability.
      </p>
      <section className="result-section">
        <SectionLabel>MODEL DETAILS</SectionLabel>
        <div className="detail-list">
          <Detail label="Model order" value="Bigram · Markov assumption" />
          <Detail label="Probability" value="Maximum likelihood + add-one" />
          <Detail label="Unknown words" value="Mapped to <UNK>" />
          <Detail label="Sentence boundaries" value="Explicit start / end markers" />
        </div>
        <p className="result-explanation">
          A bigram uses the previous token as context. Sentence markers score
          starts and endings, and unseen evaluation words are mapped to
          <code> &lt;UNK&gt; </code>.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing={`FIRST ${Math.min(result.token_probabilities.length, 40)} STEPS`}>
          NEXT-WORD PROBABILITY TRACE
        </SectionLabel>
        <div className="language-trace">
          {result.token_probabilities.slice(0, 40).map((step, index) => (
            <div className="language-trace-row" key={`${step.sentence}-${index}`}>
              <span className="trace-context">{step.previous_token}</span>
              <span className="trace-arrow">→</span>
              <strong>{step.next_token}</strong>
              <div
                className="trace-track"
                aria-label={`Probability ${step.probability.toFixed(4)}`}
              >
                <i
                  style={{
                    width: `${Math.max(2, step.probability * 100)}%`,
                  }}
                />
              </div>
              <code>{step.probability.toFixed(4)}</code>
            </div>
          ))}
        </div>
        {result.token_probabilities.length > 40 && (
          <p className="micro-note">
            Showing the first 40 of {result.token_probabilities.length} word
            transitions.
          </p>
        )}
        <p className="result-explanation">
          Each row shows the probability assigned to the next token given the
          previous token. Perplexity combines all these probabilities; one
          surprising transition can raise the overall score.
        </p>
      </section>
      <NoteBox>
        <Info size={15} />
        <span>
          Perplexity depends on tokenization, vocabulary, and corpus. It is not
          a measure of semantic quality.
        </span>
      </NoteBox>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
