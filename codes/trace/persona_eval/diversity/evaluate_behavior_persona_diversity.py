import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


DEFAULT_INPUT_PATH = "tests/data/diversity/behavior_persona.json"
DEFAULT_OUTPUT_PATH = "output/behavior_persona_diversity_results.json"
DEFAULT_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(flatten_text(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(flatten_text(v) for v in value)
    return str(value)


def load_persona_texts(path: str) -> List[str]:
    data = load_json(path)

    if isinstance(data, dict):
        personas = list(data.values())
    elif isinstance(data, list):
        personas = data
    else:
        raise ValueError("Input JSON must be either a list or a dictionary.")

    texts = []
    for persona in personas:
        if isinstance(persona, dict):
            persona = {k: v for k, v in persona.items() if k != "user_id"}
        text = flatten_text(persona).strip()
        if text:
            texts.append(text)

    if not texts:
        raise ValueError("No non-empty persona texts were found in the input file.")

    return texts


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize_l2: bool = True,
) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    if normalize_l2:
        embeddings = normalize(embeddings, norm="l2")

    return embeddings


def pairwise_cosine_average_distance(embeddings: np.ndarray) -> float:
    similarity = cosine_similarity(embeddings)
    distance = 1.0 - similarity
    upper_triangle_indices = np.triu_indices(distance.shape[0], k=1)
    return float(distance[upper_triangle_indices].mean())


def knn_average_distance(embeddings: np.ndarray, k: int) -> float:
    if embeddings.shape[0] <= k:
        raise ValueError(f"k must be smaller than the number of samples. Got k={k}.")

    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)
    return float(distances[:, 1:].mean())


def semantic_entropy(embeddings: np.ndarray, n_clusters: int, random_state: int) -> float:
    if embeddings.shape[0] < n_clusters:
        raise ValueError(
            "The number of semantic clusters must not exceed the number of samples."
        )

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    counts = np.bincount(labels, minlength=n_clusters)
    probabilities = counts / counts.sum()
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log(probabilities))
    normalized_entropy = entropy / math.log(n_clusters)
    return float(normalized_entropy)


def evaluate_diversity(args: argparse.Namespace) -> Dict[str, Any]:
    texts = load_persona_texts(args.input_path)
    model = SentenceTransformer(args.model_name, device=args.device)
    embeddings = embed_texts(model, texts, batch_size=args.batch_size)

    results = {
        "input_path": args.input_path,
        "model_name": args.model_name,
        "device": args.device,
        "num_personas": len(texts),
        "num_semantic_clusters": args.num_semantic_clusters,
        "knn_k": args.knn_k,
        "random_state": args.random_state,
        "metrics": {
            "knn_average_distance": knn_average_distance(embeddings, args.knn_k),
            "pairwise_cosine_average_distance": pairwise_cosine_average_distance(
                embeddings
            ),
            "semantic_entropy": semantic_entropy(
                embeddings,
                n_clusters=args.num_semantic_clusters,
                random_state=args.random_state,
            ),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate diversity metrics for behavior persona texts."
    )
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn_k", type=int, default=10)
    parser.add_argument("--num_semantic_clusters", type=int, default=500)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate_diversity(args)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
