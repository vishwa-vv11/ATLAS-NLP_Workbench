import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  parseCsv,
  type AnalysisResult,
  type ClassificationResult,
  type EmbeddingResult,
  type LanguageModelResult,
  type RepresentationResult,
} from "./api";
import { AppShell } from "./components/layout/AppShell";
import { WorkspaceInput } from "./features/workspaces/WorkspaceInput";
import { WorkspaceOutput } from "./features/workspaces/WorkspaceOutput";
import type { Classifier, EmbeddingMethod, Workspace } from "./types";

const SAMPLE_TEXT =
  "Natural language processing gives computers tools to work with human language. " +
  "A clear preprocessing pipeline turns raw text into useful signals.";
const SAMPLE_DOCUMENTS = [
  "Natural language processing studies human language.",
  "Language models learn patterns from text.",
  "Text classification assigns labels to documents.",
  "Tokenization splits sentences into useful units.",
].join("\n");
const SAMPLE_CSV = [
  "text,label",
  "The product is excellent and works beautifully,positive",
  "I love the quality and fast delivery,positive",
  "A wonderful experience and friendly service,positive",
  "This is a fantastic product,positive",
  "Very happy with this purchase,positive",
  "Great value and easy to use,positive",
  "The product is broken and disappointing,negative",
  "I hate the poor quality and slow delivery,negative",
  "A terrible experience with rude service,negative",
  "This is a useless product,negative",
  "Very unhappy with this purchase,negative",
  "Poor value and difficult to use,negative",
].join("\n");
const SAMPLE_EMBEDDING_TEXT = [
  "The garden has flowers, trees, green grass, and fresh soil.",
  "Flowers grow in the garden beside trees and plants.",
  "A gardener grows plants, flowers, fruit, and vegetables in soil.",
  "The orchard has fruit trees, apples, pears, and green leaves.",
  "Plants need water, sunlight, and good soil to grow.",
  "The forest has tall trees, green leaves, birds, and wildlife.",
  "Fresh vegetables grow beside herbs and flowers in the garden.",
].join("\n");
const SAMPLE_TRAINING_TEXT =
  "The cat sat on the mat. The dog sat by the door. " +
  "A cat likes to sleep. A dog likes to play.";
const SAMPLE_EVALUATION_TEXT = "The cat likes to play by the door.";

function getCsvDataset(source: string) {
  const rows = parseCsv(source);
  const headers = Object.keys(rows[0]).map((header) =>
    header.replace(/^\uFEFF/, "").toLowerCase(),
  );
  const textHeader = headers.indexOf("text");
  const labelHeader = headers.indexOf("label");
  if (textHeader < 0 || labelHeader < 0) {
    throw new Error('CSV must include columns named "text" and "label".');
  }
  const originalHeaders = Object.keys(rows[0]);
  const textColumn = originalHeaders[textHeader];
  const labelColumn = originalHeaders[labelHeader];
  const dataset = rows.map((row, index) => {
    const text = row[textColumn]?.trim();
    const label = row[labelColumn]?.trim();
    if (!text || !label) {
      throw new Error(`CSV row ${index + 2} must have non-empty text and label.`);
    }
    return { text, label };
  });
  if (dataset.length < 4) {
    throw new Error("The dataset needs at least four labeled examples.");
  }
  return dataset;
}

export default function App() {
  const [active, setActive] = useState<Workspace>("analysis");
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [mobileNavOpen, setMobileNavOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [text, setText] = useState(SAMPLE_TEXT);
  const [documents, setDocuments] = useState(SAMPLE_DOCUMENTS);
  const [csvText, setCsvText] = useState(SAMPLE_CSV);
  const [csvName, setCsvName] = useState("");
  const [classifier, setClassifier] = useState<Classifier>("naive_bayes");
  const [embeddingText, setEmbeddingText] = useState(SAMPLE_EMBEDDING_TEXT);
  const [embeddingMethod, setEmbeddingMethod] =
    useState<EmbeddingMethod>("word2vec");
  const [embeddingQuery, setEmbeddingQuery] = useState("garden");
  const [trainingText, setTrainingText] = useState(SAMPLE_TRAINING_TEXT);
  const [evaluationText, setEvaluationText] = useState(SAMPLE_EVALUATION_TEXT);

  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [representation, setRepresentation] =
    useState<RepresentationResult | null>(null);
  const [classification, setClassification] =
    useState<ClassificationResult | null>(null);
  const [neighbors, setNeighbors] = useState<EmbeddingResult | null>(null);
  const [perplexity, setPerplexity] =
    useState<LanguageModelResult | null>(null);

  useEffect(() => {
    let mounted = true;
    void api.health().then((connected) => {
      if (mounted) setApiConnected(connected);
    });
    return () => {
      mounted = false;
    };
  }, []);

  const documentsCount = useMemo(
    () => documents.split(/\r?\n/).filter((line) => line.trim()).length,
    [documents],
  );

  const runExperiment = useCallback(
    async (experiment: () => Promise<void>) => {
      setError("");
      setLoading(true);
      try {
        await experiment();
        setApiConnected(true);
      } catch (cause) {
        setError(
          cause instanceof Error
            ? cause.message
            : "The experiment could not be completed.",
        );
        void api.health().then(setApiConnected);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handleRun = () => {
    switch (active) {
      case "analysis":
        void runExperiment(async () => setAnalysis(await api.analyze(text)));
        break;
      case "represent":
        void runExperiment(async () => {
          const corpus = documents
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean);
          setRepresentation(await api.represent(corpus));
        });
        break;
      case "classify":
        void runExperiment(async () => {
          const dataset = getCsvDataset(csvText);
          setClassification(
            await api.classify(
              dataset.map((row) => row.text),
              dataset.map((row) => row.label),
              classifier,
            ),
          );
        });
        break;
      case "embeddings":
        void runExperiment(async () => {
          const corpus = embeddingText
            .split(/\r?\n/)
            .map((line) => line.trim())
            .filter(Boolean);
          setNeighbors(
            await api.embeddings(corpus, embeddingQuery.trim(), embeddingMethod),
          );
        });
        break;
      case "language":
        void runExperiment(async () => {
          setPerplexity(
            await api.languageModel(trainingText, evaluationText),
          );
        });
        break;
    }
  };

  const handleCsvUpload = async (file: File) => {
    setError("");
    if (file.size > 10 * 1024 * 1024) {
      setError("CSV file exceeds the 10 MB size limit.");
      return;
    }
    try {
      const content = await file.text();
      parseCsv(content);
      setCsvText(content);
      setCsvName(file.name);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Could not read the selected CSV file.",
      );
    }
  };

  const hasResult = {
    analysis: analysis !== null,
    represent: representation !== null,
    classify: classification !== null,
    embeddings: neighbors !== null,
    language: perplexity !== null,
  }[active];

  return (
    <AppShell
      active={active}
      apiConnected={apiConnected}
      mobileNavOpen={mobileNavOpen}
      onToggleNavigation={() => setMobileNavOpen((open) => !open)}
      onChooseWorkspace={(workspace) => {
        setActive(workspace);
        setMobileNavOpen(false);
        setError("");
      }}
    >
      <div className="workspace-grid">
        <WorkspaceInput
          active={active}
          loading={loading}
          error={error}
          onDismissError={() => setError("")}
          text={text}
          onTextChange={setText}
          documents={documents}
          documentsCount={documentsCount}
          onDocumentsChange={setDocuments}
          csvText={csvText}
          csvName={csvName}
          onCsvChange={setCsvText}
          onCsvUpload={(file) => void handleCsvUpload(file)}
          classifier={classifier}
          onClassifierChange={setClassifier}
          embeddingText={embeddingText}
          onEmbeddingTextChange={setEmbeddingText}
          embeddingMethod={embeddingMethod}
          onEmbeddingMethodChange={setEmbeddingMethod}
          embeddingQuery={embeddingQuery}
          onEmbeddingQueryChange={setEmbeddingQuery}
          trainingText={trainingText}
          onTrainingTextChange={setTrainingText}
          evaluationText={evaluationText}
          onEvaluationTextChange={setEvaluationText}
          onRun={handleRun}
        />
        <WorkspaceOutput
          active={active}
          loading={loading}
          hasResult={hasResult}
          analysis={analysis}
          representation={representation}
          classification={classification}
          neighbors={neighbors}
          embeddingMethod={embeddingMethod}
          perplexity={perplexity}
        />
      </div>
    </AppShell>
  );
}
