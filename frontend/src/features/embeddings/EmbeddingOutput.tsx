import { Info, Network } from "lucide-react";
import type { EmbeddingResult } from "../../api";
import {
  LearningSteps,
  type LearningStep,
} from "../../components/ui/LearningSteps";
import {
  EmptyState,
  NoteBox,
  SectionLabel,
} from "../../components/ui/Panel";
import type { EmbeddingMethod } from "../../types";

const METHOD_LABELS: Record<EmbeddingMethod, string> = {
  word2vec: "Word2Vec · CBOW",
  fasttext: "FastText",
  glove: "GloVe · educational",
};

const EMBEDDING_STEPS: LearningStep[] = [
  {
    title: "Tokenize the corpus",
    description:
      "The text is normalized and split into sentences and words. Those words define the model's vocabulary.",
  },
  {
    title: "Learn a vector for each word",
    description:
      "The selected model learns vectors from surrounding-word patterns (FastText also uses character pieces; GloVe fits co-occurrence counts).",
  },
  {
    title: "Compare with cosine similarity",
    description:
      "The query vector is compared with vocabulary vectors. A higher cosine score means a closer direction in the learned space.",
  },
  {
    title: "Project the space for viewing",
    description:
      "PCA compresses the learned vectors to two display dimensions. It helps reveal clusters, but can distort distances and does not change the model.",
  },
];

export function EmbeddingOutput({
  result,
  method,
}: {
  result: EmbeddingResult | null;
  method: EmbeddingMethod;
}) {
  if (!result) {
    return (
      <div className="result-body">
        <EmptyState
          title="Explore word relationships"
          hint="Choose a corpus and query to train a small embedding model."
        />
        <LearningSteps
          title="What this feature does"
          steps={EMBEDDING_STEPS}
        />
      </div>
    );
  }

  const maxScore = Math.max(
    ...result.neighbors.map(([, score]) => score),
    0.0001,
  );

  return (
    <div className="result-body">
      <div className="embedding-summary">
        <div className="embedding-symbol">
          <Network size={21} />
        </div>
        <div>
          <span>MODEL</span>
          <strong>{METHOD_LABELS[method]}</strong>
        </div>
        <div className="summary-divider" />
        <div>
          <span>VECTOR TYPE</span>
          <strong>Static · 50 dimensions</strong>
        </div>
      </div>

      <section className="result-section embedding-map-section">
        <SectionLabel trailing="2D PCA · TRAINED VECTORS">
          WORD SPACE MAP
        </SectionLabel>
        {result.projection.length > 1 ? (
          <VectorMap
            query={result.query}
            projection={result.projection}
            neighbors={result.neighbors}
          />
        ) : (
          <p className="micro-note">
            Add more distinct words to your corpus to see their relative
            positions.
          </p>
        )}
        <div className="map-legend">
          <span><i className="legend-query" /> Query word</span>
          <span><i className="legend-neighbor" /> Nearest words</span>
          <span><i className="legend-other" /> Other corpus words</span>
        </div>
        <p className="map-caption">
          Each dot is a word vector projected from 50 dimensions to 2 using
          PCA. Nearby dots have similar vectors in this corpus; the projection
          is an approximation, and the axes have no linguistic units.
        </p>
      </section>

      <section className="result-section">
        <SectionLabel trailing="COSINE SIMILARITY">NEAREST WORDS</SectionLabel>
        {result.neighbors.length ? (
          <div className="neighbor-list">
            {result.neighbors.map(([word, score], index) => (
              <div className="neighbor-row" key={word}>
                <span className="neighbor-rank">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <strong>{word}</strong>
                <div
                  className="neighbor-track"
                  aria-label={`Similarity ${score.toFixed(3)}`}
                >
                  <i
                    style={{
                      width: `${Math.max(5, (score / maxScore) * 100)}%`,
                    }}
                  />
                </div>
                <span className="neighbor-score">{score.toFixed(3)}</span>
              </div>
            ))}
          </div>
        ) : (
          <p className="micro-note">
            No other words are available in this corpus to compare with the
            query.
          </p>
        )}
        <p className="result-explanation">
          The bars compare cosine similarity scores within this result. Higher
          values indicate closer vector directions; they are not probabilities
          or proof of synonymy.
        </p>
      </section>

      <NoteBox>
        <Info size={15} />
        <span>
          Similarity reflects co-occurrence patterns in this corpus only. Small
          training samples produce unstable vectors and are for demonstration.
          In the map, the markers are a 2D projection—not a full view of all 50
          dimensions.
        </span>
      </NoteBox>
    </div>
  );
}

function VectorMap({
  query,
  projection,
  neighbors,
}: {
  query: string;
  projection: EmbeddingResult["projection"];
  neighbors: EmbeddingResult["neighbors"];
}) {
  const width = 520;
  const height = 300;
  const centerX = width / 2;
  const centerY = height / 2;
  const maxX = Math.max(...projection.map(({ x }) => Math.abs(x)), 0.0001);
  const maxY = Math.max(...projection.map(({ y }) => Math.abs(y)), 0.0001);
  const scale = Math.min(205 / maxX, 108 / maxY);
  const positions = projection.map((point) => ({
    ...point,
    px: centerX + point.x * scale,
    py: centerY - point.y * scale,
  }));
  const queryWord = projection.find(
    ({ word }) => word.toLowerCase() === query,
  )?.word;
  const queryPosition = positions.find(({ word }) => word === queryWord);
  const neighborScores = new Map(neighbors);

  return (
    <div className="vector-map">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Two-dimensional PCA projection of word vectors, highlighting ${query}`}
      >
        <line
          className="map-axis"
          x1="35"
          y1={centerY}
          x2={width - 35}
          y2={centerY}
        />
        <line
          className="map-axis"
          x1={centerX}
          y1="20"
          x2={centerX}
          y2={height - 25}
        />
        <text className="map-axis-label" x={width - 38} y={centerY - 8}>
          PC 1
        </text>
        <text className="map-axis-label" x={centerX + 8} y="19">
          PC 2
        </text>
        {queryPosition &&
          positions
            .filter(({ word }) => neighborScores.has(word))
            .map(({ word, px, py }) => (
              <line
                className="map-link"
                key={`link-${word}`}
                x1={queryPosition.px}
                y1={queryPosition.py}
                x2={px}
                y2={py}
              >
                <title>
                  {queryWord} to {word}: cosine similarity{" "}
                  {neighborScores.get(word)?.toFixed(3)}
                </title>
              </line>
            ))}
        {positions.map(({ word, px, py }) => {
          const isQuery = word === queryWord;
          const score = neighborScores.get(word);
          const isNeighbor = score !== undefined;
          const labelAnchor = px > centerX + 90 ? "end" : "start";
          const labelOffset = labelAnchor === "end" ? -9 : 9;
          return (
            <g
              className={`map-word ${isQuery ? "query" : isNeighbor ? "neighbor" : ""}`}
              key={word}
            >
              <title>
                {word}
                {score !== undefined
                  ? ` · cosine similarity ${score.toFixed(3)}`
                  : isQuery
                    ? " · query word"
                    : ""}
              </title>
              <circle cx={px} cy={py} r={isQuery ? 6 : isNeighbor ? 4.5 : 3} />
              <text
                x={px + (isQuery ? 10 : labelOffset)}
                y={py + 3}
                textAnchor={isQuery ? "start" : labelAnchor}
              >
                {word}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
