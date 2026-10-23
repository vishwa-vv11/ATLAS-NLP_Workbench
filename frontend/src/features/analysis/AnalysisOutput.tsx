import { Activity, Info } from "lucide-react";
import type { AnalysisResult } from "../../api";
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

const PIPELINE_STEPS: LearningStep[] = [
  {
    title: "Contraction expansion",
    output: null,
    description:
      "Expands supported short forms such as “can't” into separate words, making later processing more consistent.",
  },
  {
    title: "Text normalization",
    output: null,
    description:
      "Lowercases text and removes unsupported punctuation so equivalent words are handled alike.",
  },
  {
    title: "Sentence splitting",
    output: null,
    description:
      "Finds sentence boundaries so later steps can work with sentence-sized units.",
  },
  {
    title: "Word tokenization",
    output: null,
    description:
      "Breaks sentences into individual word tokens, the units counted and modeled downstream.",
  },
  {
    title: "Stop-word filtering",
    output: null,
    description:
      "Removes common grammatical words while preserving negation such as “not”, which can change meaning.",
  },
  {
    title: "Stemming",
    output: null,
    description:
      "Shortens related word forms to approximate roots; the result may not be a dictionary word.",
  },
  {
    title: "Lemmatization",
    output: null,
    description:
      "Maps tokens to dictionary forms where possible. This project uses noun lemmatization by default.",
  },
];

const ANALYSIS_LEARNING_STEPS: LearningStep[] = [
  {
    title: "Top terms are counts",
    description:
      "Longer bars mean a word appeared more often after stop-word filtering; they do not indicate importance by themselves.",
  },
  {
    title: "Keywords are a TF-IDF baseline",
    description:
      "The score ranks terms in this single passage. It is a simple teaching aid, not a trained keyword model.",
  },
  {
    title: "Sentiment is a tiny lexicon check",
    description:
      "A small list of positive and negative words contributes to the score. Negation, context, sarcasm, and domain language are not reliably understood.",
  },
];

export function AnalysisOutput({ result }: { result: AnalysisResult | null }) {
  if (!result) {
    return (
      <div className="result-body">
        <EmptyState
          title="Your results will appear here"
          hint="Add source text and run an analysis to inspect the pipeline."
        />
        <LearningSteps
          title="What this feature does"
          steps={ANALYSIS_LEARNING_STEPS}
        />
      </div>
    );
  }

  const frequencies = Object.entries(result.word_frequency).slice(0, 5);
  const highestFrequency = Math.max(
    1,
    ...Object.values(result.word_frequency),
  );
  const pipelineOutputs = [
    result.preprocessing.expanded_text,
    result.preprocessing.cleaned_text,
    result.preprocessing.sentences.join(" / "),
    result.preprocessing.tokens.join(" · "),
    result.preprocessing.without_stop_words.join(" · "),
    result.preprocessing.stemmed_tokens.join(" · "),
    result.preprocessing.lemmatized_tokens.join(" · "),
  ];
  const pipelineSteps = PIPELINE_STEPS.map((step, index) => ({
    ...step,
    output: (
      <code title={pipelineOutputs[index]}>
        {formatPipelineOutput(pipelineOutputs[index]) || "—"}
      </code>
    ),
  }));

  return (
    <div className="result-body">
      <div className="metric-row">
        <Metric label="SENTENCES" value={result.preprocessing.sentences.length} />
        <Metric label="TOKENS" value={result.preprocessing.tokens.length} />
        <Metric
          label="AFTER FILTER"
          value={result.preprocessing.without_stop_words.length}
          accent
        />
      </div>

      <section className="result-section">
        <SectionLabel trailing="07 STEPS">PREPROCESSING PIPELINE</SectionLabel>
        <LearningSteps title="Each step: output, then explanation" steps={pipelineSteps} />
      </section>

      <div className="result-section insights-grid">
        <section>
          <SectionLabel>TOP TERMS</SectionLabel>
          {frequencies.length ? (
            <div className="term-list">
              {frequencies.map(([term, count]) => (
                <div className="term-row" key={term}>
                  <span>{term}</span>
                  <div className="term-bar">
                    <i
                      style={{
                        width: `${Math.max(10, (count / highestFrequency) * 100)}%`,
                      }}
                    />
                  </div>
                  <b>{count}</b>
                </div>
              ))}
            </div>
          ) : (
            <p className="micro-note">No terms remain after filtering.</p>
          )}
          <p className="result-explanation">
            Counts are calculated after stop-word filtering. Bar length compares
            each displayed word with the most frequent displayed word.
          </p>
        </section>

        <section>
          <SectionLabel>
            SENTIMENT BASELINE <Info size={13} aria-label="Educational baseline" />
          </SectionLabel>
          <div className="sentiment-card">
            <span
              className={`sentiment-mark ${result.sentiment_baseline.label}`}
              aria-hidden="true"
            >
              <Activity size={16} />
            </span>
            <div>
              <strong>{capitalize(result.sentiment_baseline.label)}</strong>
              <span>
                Lexicon score:{" "}
                {result.sentiment_baseline.score > 0 ? "+" : ""}
                {result.sentiment_baseline.score}
              </span>
            </div>
            <span className="baseline-tag">BASIC</span>
          </div>
          <p className="micro-note">
            Educational lexicon baseline — not a validated sentiment model.
          </p>
        </section>
      </div>

      <section className="keyword-row">
        <SectionLabel>KEYWORDS</SectionLabel>
        <div className="keyword-list">
          {result.keywords.slice(0, 5).map(([word, score]) => (
            <span className="keyword" key={word}>
              {word}
              <b>{score.toFixed(2)}</b>
            </span>
          ))}
        </div>
      </section>
      <p className="result-explanation">
        Scores rank terms in this passage using a simple TF-IDF baseline; they
        are not a trained keyword model.
      </p>
      <NoteBox>
        <Info size={15} />
        <span>
          Results are simple classical baselines. Review the method notes before
          interpreting them as evidence about a larger dataset.
        </span>
      </NoteBox>
    </div>
  );
}

function formatPipelineOutput(value: string) {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > 360
    ? `${normalized.slice(0, 360)}…`
    : normalized;
}

function capitalize(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
