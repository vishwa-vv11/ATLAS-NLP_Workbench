import type { LucideIcon } from "lucide-react";

export type Workspace =
  | "analysis"
  | "represent"
  | "classify"
  | "embeddings"
  | "language";

export interface WorkspaceItem {
  id: Workspace;
  label: string;
  shortLabel: string;
  icon: LucideIcon;
  group: "Explore" | "Evaluate";
}

export type Classifier = "naive_bayes" | "logistic_regression";
export type EmbeddingMethod = "word2vec" | "fasttext" | "glove";
