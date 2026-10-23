export interface AnalysisResult {
  preprocessing: {
    expanded_text: string;
    cleaned_text: string;
    sentences: string[];
    tokens: string[];
    without_stop_words: string[];
    stemmed_tokens: string[];
    lemmatized_tokens: string[];
  };
  word_frequency: Record<string, number>;
  keywords: [string, number][];
  sentiment_baseline: { label: string; score: number };
}

export interface ClassificationResult {
  test_texts: string[];
  true_labels: string[];
  predictions: string[];
  metrics: {
    accuracy: number;
    precision_weighted: number;
    recall_weighted: number;
    f1_weighted: number;
    labels: string[];
    confusion_matrix: number[][];
  };
  errors: { text: string; true_label: string; predicted_label: string }[];
}

export interface RepresentationResult {
  vocabulary: string[];
  one_hot_first_document: Record<string, number[]>;
  bag_of_words: number[][];
  ngrams: string[][][];
  tfidf_vocabulary: string[];
  tfidf: number[][];
}

export interface LanguageModelResult {
  perplexity: number;
  training_sentences: number;
  evaluation_sentences: number;
}

export interface EmbeddingResult {
  query: string;
  neighbors: [string, number][];
  projection: { word: string; x: number; y: number }[];
}

async function post<T>(path: string, payload: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new Error(
      "Could not reach the NLP API. Start the Python API and try again.",
    );
  }

  const result: unknown = await response.json();
  if (!response.ok) {
    const detail =
      typeof result === "object" &&
      result !== null &&
      "detail" in result
        ? result.detail
        : undefined;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map((item) =>
                typeof item === "object" && item !== null && "msg" in item
                  ? item.msg
                  : "",
              )
              .filter((item): item is string => typeof item === "string" && !!item)
              .join("; ")
          : "The request could not be completed.";
    throw new Error(message);
  }
  return result as T;
}

export const api = {
  health: async (): Promise<boolean> => {
    try {
      const response = await fetch("/api/health");
      return response.ok;
    } catch {
      return false;
    }
  },
  analyze: (text: string) => post<AnalysisResult>("/api/analyze", { text }),
  represent: (documents: string[], ngram_size = 2) =>
    post<RepresentationResult>("/api/represent", { documents, ngram_size }),
  classify: (
    texts: string[],
    labels: string[],
    model: "naive_bayes" | "logistic_regression",
  ) =>
    post<ClassificationResult>("/api/classify", {
      texts,
      labels,
      model,
    }),
  embeddings: (
    texts: string[],
    query: string,
    method: "word2vec" | "fasttext" | "glove",
  ) =>
    post<EmbeddingResult>(
      "/api/embeddings",
      {
        texts,
        query,
        method,
        top_n: 8,
        vector_size: 50,
        include_projection: true,
      },
    ),
  languageModel: (training_text: string, evaluation_text: string) =>
    post<LanguageModelResult>("/api/language-model", {
      training_text,
      evaluation_text,
      alpha: 1,
    }),
};

export function parseCsv(source: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quoted) {
      if (char === '"' && source[index + 1] === '"') {
        cell += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        cell += char;
      }
    } else if (char === '"' && cell.length === 0) {
      quoted = true;
    } else if (char === ",") {
      row.push(cell);
      cell = "";
    } else if (char === "\n" || char === "\r") {
      row.push(cell);
      if (row.some((value) => value.trim())) rows.push(row);
      row = [];
      cell = "";
      if (char === "\r" && source[index + 1] === "\n") index += 1;
    } else {
      cell += char;
    }
  }
  if (quoted) throw new Error("The CSV contains an unclosed quoted field.");
  row.push(cell);
  if (row.some((value) => value.trim())) rows.push(row);
  if (rows.length < 2) {
    throw new Error("CSV needs a header row and at least one data row.");
  }

  const headers = rows[0].map((header) => header.trim());
  if (new Set(headers).size !== headers.length || headers.some((h) => !h)) {
    throw new Error("CSV headers must be non-empty and unique.");
  }
  return rows.slice(1).map((values, rowIndex) => {
    if (values.length !== headers.length) {
      throw new Error(`CSV row ${rowIndex + 2} has an unexpected column count.`);
    }
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index].trim()]),
    );
  });
}
