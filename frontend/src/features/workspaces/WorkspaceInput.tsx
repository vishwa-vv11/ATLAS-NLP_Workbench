import {
  Check,
  Database,
  FileText,
  FlaskConical,
  Info,
  Play,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import type { ReactNode } from "react";
import { Panel, PanelHeading } from "../../components/ui/Panel";
import type { Classifier, EmbeddingMethod, Workspace } from "../../types";

interface WorkspaceInputProps {
  active: Workspace;
  loading: boolean;
  error: string;
  onDismissError: () => void;
  text: string;
  onTextChange: (value: string) => void;
  documents: string;
  documentsCount: number;
  onDocumentsChange: (value: string) => void;
  csvText: string;
  csvName: string;
  onCsvChange: (value: string) => void;
  onCsvUpload: (file: File) => void;
  classifier: Classifier;
  onClassifierChange: (value: Classifier) => void;
  embeddingText: string;
  onEmbeddingTextChange: (value: string) => void;
  embeddingMethod: EmbeddingMethod;
  onEmbeddingMethodChange: (value: EmbeddingMethod) => void;
  embeddingQuery: string;
  onEmbeddingQueryChange: (value: string) => void;
  trainingText: string;
  onTrainingTextChange: (value: string) => void;
  evaluationText: string;
  onEvaluationTextChange: (value: string) => void;
  onRun: () => void;
}

export function WorkspaceInput(props: WorkspaceInputProps) {
  const heading = {
    analysis: "Source text",
    represent: "Document collection",
    classify: "Labeled dataset",
    embeddings: "Training corpus",
    language: "Model configuration",
  }[props.active];

  return (
    <Panel className="input-panel">
      <PanelHeading
        icon={<FileText size={17} />}
        title={heading}
        trailing={
          props.active === "classify" ? (
            <span className="required-tag">CSV · TEXT, LABEL</span>
          ) : (
            <span className="input-mode">INPUT</span>
          )
        }
      />
      {props.active === "analysis" && <AnalysisInput {...props} />}
      {props.active === "represent" && <RepresentationInput {...props} />}
      {props.active === "classify" && <ClassificationInput {...props} />}
      {props.active === "embeddings" && <EmbeddingInput {...props} />}
      {props.active === "language" && <LanguageInput {...props} />}
      {props.error && (
        <div className="error-message" role="alert">
          <X size={15} />
          <span>{props.error}</span>
          <button
            onClick={props.onDismissError}
            aria-label="Dismiss error"
            type="button"
          >
            <X size={14} />
          </button>
        </div>
      )}
    </Panel>
  );
}

function AnalysisInput({
  loading,
  text,
  onTextChange,
  onRun,
}: WorkspaceInputProps) {
  return (
    <>
      <p className="panel-description">
        Paste a passage to inspect how the preprocessing pipeline transforms it.
      </p>
      <textarea
        className="text-editor"
        value={text}
        onChange={(event) => onTextChange(event.target.value)}
        maxLength={20_000}
        aria-label="Text to analyze"
      />
      <div className="editor-footer">
        <span>
          <span className="editor-dot" /> Plain text
        </span>
        <span>{text.length.toLocaleString()} / 20,000</span>
      </div>
      <div className="panel-actions">
        <span className="quiet-note">
          <Sparkles size={14} /> No generative models
        </span>
        <RunButton loading={loading} disabled={!text.trim()} onClick={onRun}>
          Run analysis <span className="button-key">↵</span>
        </RunButton>
      </div>
    </>
  );
}

function RepresentationInput({
  loading,
  documents,
  documentsCount,
  onDocumentsChange,
  onRun,
}: WorkspaceInputProps) {
  return (
    <>
      <p className="panel-description">
        One document per line. The vocabulary is learned from this corpus.
      </p>
      <textarea
        className="text-editor multi-editor"
        value={documents}
        onChange={(event) => onDocumentsChange(event.target.value)}
        maxLength={20_000}
        aria-label="Documents to represent"
      />
      <div className="editor-footer">
        <span>
          <span className="editor-dot" /> {documentsCount} documents
        </span>
        <span>One document per line</span>
      </div>
      <div className="panel-actions">
        <span className="quiet-note">
          <Database size={14} /> Manual TF–IDF
        </span>
        <RunButton
          loading={loading}
          disabled={!documentsCount}
          onClick={onRun}
        >
          Build features
        </RunButton>
      </div>
    </>
  );
}

function ClassificationInput({
  loading,
  csvText,
  csvName,
  onCsvChange,
  onCsvUpload,
  classifier,
  onClassifierChange,
  onRun,
}: WorkspaceInputProps) {
  return (
    <>
      <p className="panel-description">
        Upload a labeled CSV or edit the small sample. Keep test text out of
        model fitting.
      </p>
      <label className={`upload-zone ${csvName ? "uploaded" : ""}`}>
        {csvName ? <Check size={18} /> : <Upload size={18} />}
        <span>
          <strong>{csvName || "Choose a CSV file"}</strong>
          <small>UTF-8 · 10 MB max · text and label columns</small>
        </span>
        <input
          type="file"
          accept=".csv,text/csv"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onCsvUpload(file);
          }}
        />
        <span className="upload-action">{csvName ? "Replace" : "Browse"}</span>
      </label>
      <textarea
        className="text-editor csv-editor"
        value={csvText}
        onChange={(event) => onCsvChange(event.target.value)}
        aria-label="CSV dataset"
        spellCheck={false}
      />
      <div className="field-row">
        <label className="field-label">
          Classifier
          <select
            value={classifier}
            onChange={(event) =>
              onClassifierChange(event.target.value as Classifier)
            }
          >
            <option value="naive_bayes">Multinomial Naive Bayes</option>
            <option value="logistic_regression">Logistic Regression</option>
          </select>
        </label>
        <div className="field-label">
          Split
          <div className="static-field">
            80% train <span>·</span> 20% test
          </div>
        </div>
      </div>
      <div className="panel-actions">
        <span className="quiet-note">
          <Database size={14} /> Data stays in this session
        </span>
        <RunButton loading={loading} disabled={!csvText.trim()} onClick={onRun}>
          Train &amp; evaluate
        </RunButton>
      </div>
    </>
  );
}

function EmbeddingInput({
  loading,
  embeddingText,
  onEmbeddingTextChange,
  embeddingMethod,
  onEmbeddingMethodChange,
  embeddingQuery,
  onEmbeddingQueryChange,
  onRun,
}: WorkspaceInputProps) {
  return (
    <>
      <p className="panel-description">
        Train a small static embedding on your corpus, then explore its nearest
        words.
      </p>
      <textarea
        className="text-editor multi-editor"
        value={embeddingText}
        onChange={(event) => onEmbeddingTextChange(event.target.value)}
        aria-label="Embedding training corpus"
      />
      <div className="field-row">
        <label className="field-label">
          Method
          <select
            value={embeddingMethod}
            onChange={(event) =>
              onEmbeddingMethodChange(event.target.value as EmbeddingMethod)
            }
          >
            <option value="word2vec">Word2Vec · CBOW</option>
            <option value="fasttext">FastText</option>
            <option value="glove">GloVe · educational</option>
          </select>
        </label>
        <label className="field-label">
          Query word
          <input
            value={embeddingQuery}
            onChange={(event) => onEmbeddingQueryChange(event.target.value)}
            placeholder="e.g. garden"
          />
        </label>
      </div>
      <div className="panel-actions">
        <span className="quiet-note">
          <Info size={14} /> Small-corpus demo
        </span>
        <RunButton
          loading={loading}
          disabled={!embeddingText.trim() || !embeddingQuery.trim()}
          onClick={onRun}
        >
          Train &amp; explore
        </RunButton>
      </div>
    </>
  );
}

function LanguageInput({
  loading,
  trainingText,
  onTrainingTextChange,
  evaluationText,
  onEvaluationTextChange,
  onRun,
}: WorkspaceInputProps) {
  return (
    <>
      <p className="panel-description">
        Fit an add-one-smoothed bigram model and score unseen text.
      </p>
      <label className="field-label block-label">
        Training corpus
        <textarea
          className="text-editor language-editor"
          value={trainingText}
          onChange={(event) => onTrainingTextChange(event.target.value)}
          aria-label="Training corpus"
        />
      </label>
      <label className="field-label block-label">
        Held-out evaluation
        <textarea
          className="text-editor language-editor eval-editor"
          value={evaluationText}
          onChange={(event) => onEvaluationTextChange(event.target.value)}
          aria-label="Held out evaluation text"
        />
      </label>
      <div className="panel-actions">
        <span className="quiet-note">
          <FlaskConical size={14} /> Laplace α = 1
        </span>
        <RunButton
          loading={loading}
          disabled={!trainingText.trim() || !evaluationText.trim()}
          onClick={onRun}
        >
          Calculate perplexity
        </RunButton>
      </div>
    </>
  );
}

function RunButton({
  children,
  loading,
  disabled,
  onClick,
}: {
  children: ReactNode;
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      className="primary-button"
      onClick={onClick}
      disabled={loading || disabled}
      type="button"
    >
      <Play size={14} fill="currentColor" />
      {children}
    </button>
  );
}
