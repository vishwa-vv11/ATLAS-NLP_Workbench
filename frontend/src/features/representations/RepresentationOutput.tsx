import { Info } from "lucide-react";
import type { RepresentationResult } from "../../api";
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

const REPRESENTATION_STEPS: LearningStep[] = [
  {
    title: "Build one shared vocabulary",
    description:
      "The distinct tokens across your documents become the columns used to describe every document.",
  },
  {
    title: "Encode presence and counts",
    description:
      "One-hot vectors mark whether a term appears in the first document; Bag of Words counts each term per document.",
  },
  {
    title: "Inspect word sequences",
    description:
      "N-grams group neighboring tokens (the current workflow builds bigrams) to retain short local phrases.",
  },
  {
    title: "Weight terms with TF-IDF",
    description:
      "Term frequency is reduced for words found in many documents, then each document vector is L2-normalized.",
  },
  {
    title: "Read the matrix",
    description:
      "Rows are documents and columns are vocabulary terms. A larger value means a stronger feature under that representation.",
  },
];

export function RepresentationOutput({
  result,
}: {
  result: RepresentationResult | null;
}) {
  if (!result) {
    return (
      <div className="result-body">
        <EmptyState
          title="No feature matrix yet"
          hint="Enter a few documents to build vocabulary, counts, n-grams, and TF–IDF."
        />
        <LearningSteps
          title="What this feature does"
          steps={REPRESENTATION_STEPS}
        />
      </div>
    );
  }

  return (
    <div className="result-body">
      <div className="metric-row">
        <Metric label="DOCUMENTS" value={result.bag_of_words.length} />
        <Metric label="VOCABULARY" value={result.vocabulary.length} />
        <Metric label="FEATURES" value={result.tfidf[0]?.length ?? 0} accent />
      </div>
      <section className="result-section">
        <SectionLabel trailing="FIRST-SEEN ORDER">VOCABULARY</SectionLabel>
        <div className="vocabulary-list">
          {result.vocabulary.map((word, index) => (
            <span className="vocabulary-item" key={word}>
              <i>{String(index + 1).padStart(2, "0")}</i>
              {word}
            </span>
          ))}
        </div>
        <p className="result-explanation">
          Each distinct token becomes a feature column. The list shows the order
          in which terms first appeared.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing="DOCUMENT × TERM COUNTS">
          BAG OF WORDS
        </SectionLabel>
        <Matrix table={result.bag_of_words} columns={result.vocabulary} />
        <p className="result-explanation">
          Each row is a document; each column is a vocabulary term. The cell
          value is the number of times that term appears in that document.
        </p>
        <SectionLabel trailing="FIRST DOCUMENT">ONE-HOT PRESENCE</SectionLabel>
        <div className="one-hot-list">
          {Object.entries(result.one_hot_first_document).map(
            ([term, vector]) => (
              <span className={vector[0] ? "present" : ""} key={term}>
                <b>{term}</b>
                <code>{vector.join(" ")}</code>
              </span>
            ),
          )}
        </div>
        <p className="result-explanation">
          Each token is represented by a vector with one 1 at its vocabulary
          position and 0s elsewhere. Unlike Bag of Words, this encodes a token
          identity rather than document-level word counts.
        </p>
        <SectionLabel trailing="NEIGHBORING TOKEN PAIRS">BIGRAMS</SectionLabel>
        <div className="ngram-list">
          {result.ngrams.map((documentNgrams, index) => (
            <div key={index}>
              <span>D{String(index + 1).padStart(2, "0")}</span>
              <code>
                {documentNgrams.length
                  ? documentNgrams.map((ngram) => ngram.join(" ")).join(" · ")
                  : "No adjacent token pairs"}
              </code>
            </div>
          ))}
        </div>
        <p className="result-explanation">
          A bigram joins each token to the next one and preserves local word
          order, unlike a simple term-count matrix.
        </p>
      </section>
      <section className="result-section">
        <SectionLabel trailing="SMOOTHED IDF · L2 NORMALIZED">
          TF–IDF WEIGHTS
        </SectionLabel>
        <Matrix
          table={result.tfidf}
          columns={result.tfidf_vocabulary}
          decimals
        />
        <p className="result-explanation">
          TF-IDF boosts terms that help distinguish a document and downweights
          terms common across the collection. Values are L2-normalized per
          document.
        </p>
        <SectionLabel trailing="FIRST DOCUMENT · TERM-LEVEL CALCULATION">
          HOW TF–IDF IS COMPUTED
        </SectionLabel>
        <div className="matrix-scroll">
          <table className="data-matrix tfidf-calculation">
            <thead>
              <tr>
                <th>TERM</th>
                <th>TF</th>
                <th>DOC FREQ</th>
                <th>IDF</th>
                <th>TF × IDF</th>
                <th>NORMALIZED</th>
              </tr>
            </thead>
            <tbody>
              {result.vocabulary.map((term, index) => {
                const termFrequency = result.bag_of_words[0]?.[index] ?? 0;
                if (!termFrequency) return null;
                const documentFrequency = result.bag_of_words.filter(
                  (row) => (row[index] ?? 0) > 0,
                ).length;
                const inverseDocumentFrequency =
                  Math.log(
                    (1 + result.bag_of_words.length) /
                      (1 + documentFrequency),
                  ) + 1;
                const rawWeight = termFrequency * inverseDocumentFrequency;
                return (
                  <tr key={term}>
                    <td>{term}</td>
                    <td>{termFrequency}</td>
                    <td>{documentFrequency}</td>
                    <td>{inverseDocumentFrequency.toFixed(3)}</td>
                    <td>{rawWeight.toFixed(3)}</td>
                    <td>{(result.tfidf[0]?.[index] ?? 0).toFixed(3)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="result-explanation">
          IDF = log((1 + document count) / (1 + documents containing the term))
          + 1. TF × IDF gives the raw weight; the final column is that weight
          after L2 normalization across the document.
        </p>
      </section>
      <NoteBox>
        <Info size={15} />
        <span>
          Terms present in every document receive lower IDF weight. This manual
          implementation is intended for learning and small collections.
        </span>
      </NoteBox>
    </div>
  );
}
