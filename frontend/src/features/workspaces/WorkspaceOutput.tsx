import { Activity } from "lucide-react";
import {
  type AnalysisResult,
  type ClassificationResult,
  type EmbeddingResult,
  type LanguageModelResult,
  type RepresentationResult,
} from "../../api";
import { Panel, PanelHeading, LoadingState } from "../../components/ui/Panel";
import { AnalysisOutput } from "../analysis/AnalysisOutput";
import { ClassificationOutput } from "../classification/ClassificationOutput";
import { EmbeddingOutput } from "../embeddings/EmbeddingOutput";
import { LanguageOutput } from "../language/LanguageOutput";
import { RepresentationOutput } from "../representations/RepresentationOutput";
import type { EmbeddingMethod, Workspace } from "../../types";

export function WorkspaceOutput({
  active,
  loading,
  hasResult,
  analysis,
  representation,
  classification,
  neighbors,
  embeddingMethod,
  perplexity,
}: {
  active: Workspace;
  loading: boolean;
  hasResult: boolean;
  analysis: AnalysisResult | null;
  representation: RepresentationResult | null;
  classification: ClassificationResult | null;
  neighbors: EmbeddingResult | null;
  embeddingMethod: EmbeddingMethod;
  perplexity: LanguageModelResult | null;
}) {
  return (
    <Panel className="results-panel">
      <PanelHeading
        icon={<Activity size={17} />}
        title="Experiment output"
        trailing={
          <span
            className={`result-status ${loading ? "running" : hasResult ? "complete" : ""}`}
          >
            <span />
            {loading ? "RUNNING" : hasResult ? "COMPLETE" : "READY"}
          </span>
        }
      />
      {loading ? (
        <LoadingState />
      ) : (
        <OutputContent
          active={active}
          analysis={analysis}
          representation={representation}
          classification={classification}
          neighbors={neighbors}
          embeddingMethod={embeddingMethod}
          perplexity={perplexity}
        />
      )}
    </Panel>
  );
}

function OutputContent({
  active,
  analysis,
  representation,
  classification,
  neighbors,
  embeddingMethod,
  perplexity,
}: Omit<Parameters<typeof WorkspaceOutput>[0], "loading" | "hasResult">) {
  switch (active) {
    case "analysis":
      return <AnalysisOutput result={analysis} />;
    case "represent":
      return <RepresentationOutput result={representation} />;
    case "classify":
      return <ClassificationOutput result={classification} />;
    case "embeddings":
      return (
        <EmbeddingOutput result={neighbors} method={embeddingMethod} />
      );
    case "language":
      return <LanguageOutput result={perplexity} />;
  }
}
