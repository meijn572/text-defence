# -*- coding: utf-8 -*-
"""Run classic CPU-friendly baselines for spam/adversarial text detection.

Baselines:
- GAS-lite graph co-occurrence baseline
- Word2Vec-w + LR
- Word2Vec-c + LR
- Word2Vec-c + GBDT
- Doc2Vec-c + GBDT

The implementation uses character-level tokenization to match the "-c" setting.
Sentence vectors are produced by IDF-weighted pooling over character embeddings,
which is a lightweight substitute for the self-attention pooling described in the
baseline family and is practical for CPU reproduction.

GAS-lite is a CPU-friendly approximation of GAS: it builds a character
co-occurrence graph from historical training texts, propagates spam risk over the
graph, then aggregates graph-derived character statistics as sentence features.
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning)

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data" / "adversarial"
RESULT_DIR = BASE / "results"
RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
VECTOR_SIZE = 100
MIN_COUNT = 1
EPOCHS = 20
GAS_MAX_VOCAB = 3000
GAS_WINDOW = 4
GAS_PROP_STEPS = 12
GAS_ALPHA = 0.85


def char_tokens(text: object) -> list[str]:
    text = "" if pd.isna(text) else str(text)
    return [ch for ch in text.strip() if not ch.isspace()]


def word_tokens(text: object) -> list[str]:
    import jieba

    text = "" if pd.isna(text) else str(text)
    return [token.strip() for token in jieba.lcut(text) if token.strip()]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_filename = os.environ.get("TRAIN_FILE", "train.csv")
    train_df = pd.read_csv(DATA_DIR / train_filename)
    test_df = pd.read_csv(DATA_DIR / "test_full.csv")
    train_df = train_df.dropna(subset=["text", "label"]).copy()
    test_df = test_df.dropna(subset=["text", "label"]).copy()
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)
    return train_df, test_df


def build_idf(texts: list[str], analyzer=char_tokens) -> dict[str, float]:
    vectorizer = TfidfVectorizer(analyzer=analyzer, lowercase=False)
    vectorizer.fit(texts)
    return dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))


def w2v_sentence_vectors(model, tokenized_texts: list[list[str]], idf: dict[str, float]) -> np.ndarray:
    vectors = np.zeros((len(tokenized_texts), model.vector_size), dtype=np.float32)
    for row, tokens in enumerate(tokenized_texts):
        weighted = []
        weights = []
        for token in tokens:
            if token in model.wv:
                weight = idf.get(token, 1.0)
                weighted.append(model.wv[token] * weight)
                weights.append(weight)
        if weighted:
            vectors[row] = np.sum(weighted, axis=0) / max(float(np.sum(weights)), 1e-8)
    return vectors


def metric_row(model_name: str, split_name: str, y_true: list[int], y_pred: np.ndarray, elapsed: float) -> dict[str, object]:
    return {
        "model": model_name,
        "attack_type": split_name,
        "n_samples": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "seconds": elapsed,
    }


def evaluation_groups(test_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    """返回标签比例一致的评测子集。

    test_full.csv 是完整数据池，A-I/J-K-L 行本身只有攻击垃圾样本；
    直接按 attack_type 切分会让攻击子集没有正常类，造成 Precision/F1 虚高。
    因此攻击子集优先读取对应的混合文件（全部正常 + 对应攻击垃圾）。
    """
    groups = [("Original", test_df["attack_type"].isna())]
    file_map = {
        "A": "adv_A_char_delete.csv",
        "B": "adv_B_char_insert.csv",
        "C": "adv_C_homoglyph_unicode.csv",
        "D": "adv_D_zero_width.csv",
        "E": "adv_E_synonym.csv",
        "F": "adv_F_homophone_cn.csv",
        "G": "adv_G_homoglyph_cn.csv",
        "H": "adv_H_fanjian_split.csv",
        "I": "adv_I_char_shuffle.csv",
        "J": "adv_J_strong_shuffle.csv",
        "K": "adv_K_strong_homophone.csv",
        "L": "adv_L_combined.csv",
        "M": "adv_M_pinyin_abbrev.csv",
    }
    for attack_id, filename in file_map.items():
        path = DATA_DIR / filename
        if path.exists():
            group_df = pd.read_csv(path).dropna(subset=["text", "label"])
            groups.append((attack_id, group_df))
        else:
            groups.append((attack_id, test_df["attack_type"] == attack_id))
    return groups


def evaluate_by_attack(
    model_name: str,
    clf,
    x_test: np.ndarray,
    test_df: pd.DataFrame,
    group_feature_builder=None,
) -> list[dict[str, object]]:
    start = time.perf_counter()
    y_pred_all = clf.predict(x_test)
    elapsed = time.perf_counter() - start

    rows = []
    groups = evaluation_groups(test_df)

    y_true_all = test_df["label"].tolist()
    rows.append(metric_row(model_name, "ALL", y_true_all, y_pred_all, elapsed))
    for name, group in groups:
        if isinstance(group, pd.DataFrame):
            group_texts = group["text"].astype(str).tolist()
            group_labels = group["label"].astype(int).tolist()
            # 该子集需要重新预测，不能复用 test_full 的位置索引。
            if group_feature_builder is None:
                raise ValueError("混合攻击子集需要 group_feature_builder")
            group_features = group_feature_builder(group_texts)
            group_pred = clf.predict(group_features)
        else:
            mask = group
            if int(mask.sum()) == 0:
                continue
            idx = np.where(mask.to_numpy())[0]
            group_labels = test_df.iloc[idx]["label"].tolist()
            group_pred = y_pred_all[idx]
        if not group_labels:
            continue
        rows.append(metric_row(model_name, name, group_labels, group_pred, elapsed))
    return rows


def build_gas_vocab(tokenized_texts: list[list[str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for tokens in tokenized_texts:
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    most_common = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:GAS_MAX_VOCAB]
    return {token: idx for idx, (token, _) in enumerate(most_common)}


def build_gas_transition(tokenized_texts: list[list[str]], vocab: dict[str, int]) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    n_vocab = len(vocab)
    edge_weights: dict[tuple[int, int], float] = {}

    for tokens in tokenized_texts:
        ids = [vocab[token] for token in tokens if token in vocab]
        for pos, src in enumerate(ids):
            end = min(len(ids), pos + GAS_WINDOW + 1)
            for dst in ids[pos + 1:end]:
                if src == dst:
                    continue
                edge_weights[(src, dst)] = edge_weights.get((src, dst), 0.0) + 1.0
                edge_weights[(dst, src)] = edge_weights.get((dst, src), 0.0) + 1.0

    for (src, dst), weight in edge_weights.items():
        rows.append(src)
        cols.append(dst)
        data.append(weight)

    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(n_vocab, n_vocab), dtype=np.float32)
    matrix = matrix + sparse.eye(n_vocab, dtype=np.float32, format="csr")
    row_sum = np.asarray(matrix.sum(axis=1)).ravel()
    row_sum[row_sum == 0] = 1.0
    inv_degree = sparse.diags(1.0 / row_sum, format="csr")
    return inv_degree @ matrix


def gas_prior_scores(tokenized_texts: list[list[str]], labels: np.ndarray, vocab: dict[str, int]) -> np.ndarray:
    spam_counts = np.ones(len(vocab), dtype=np.float32)
    total_counts = np.full(len(vocab), 2.0, dtype=np.float32)
    for tokens, label in zip(tokenized_texts, labels):
        seen = {vocab[token] for token in tokens if token in vocab}
        for idx in seen:
            total_counts[idx] += 1.0
            spam_counts[idx] += float(label)
    return spam_counts / total_counts


def gas_features(tokenized_texts: list[list[str]], vocab: dict[str, int], risk: np.ndarray, idf: dict[str, float]) -> np.ndarray:
    features = np.zeros((len(tokenized_texts), 10), dtype=np.float32)
    for row, tokens in enumerate(tokenized_texts):
        ids = [vocab[token] for token in tokens if token in vocab]
        if not ids:
            continue
        scores = risk[ids]
        weights = np.asarray([idf.get(tokens[i], 1.0) for i, token in enumerate(tokens) if token in vocab], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        unique_ids = np.unique(ids)
        features[row] = np.asarray([
            float(scores.mean()),
            float(scores.max()),
            float(scores.min()),
            float(scores.std()),
            float(np.quantile(scores, 0.25)),
            float(np.quantile(scores, 0.75)),
            float((scores * weights).sum()),
            float((scores > 0.6).mean()),
            float(len(unique_ids) / max(len(ids), 1)),
            float(len(ids)),
        ], dtype=np.float32)
    return features


def run_gas_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, object]]:
    print("[GAS-lite] building character co-occurrence graph...")
    train_tokens = [char_tokens(t) for t in train_df["text"].tolist()]
    test_tokens = [char_tokens(t) for t in test_df["text"].tolist()]
    y_train = train_df["label"].to_numpy()
    idf = build_idf(train_df["text"].astype(str).tolist())

    start = time.perf_counter()
    vocab = build_gas_vocab(train_tokens)
    transition = build_gas_transition(train_tokens, vocab)
    prior = gas_prior_scores(train_tokens, y_train, vocab)
    risk = prior.copy()
    for _ in range(GAS_PROP_STEPS):
        risk = GAS_ALPHA * transition.dot(risk) + (1.0 - GAS_ALPHA) * prior
    print(f"[GAS-lite] graph propagation done in {time.perf_counter() - start:.1f}s | vocab={len(vocab)}")

    x_train = gas_features(train_tokens, vocab, risk, idf)
    x_test = gas_features(test_tokens, vocab, risk, idf)

    print("[GAS-lite] training classifier...")
    start = time.perf_counter()
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(x_train, y_train)
    print(f"[GAS-lite] classifier training done in {time.perf_counter() - start:.1f}s")
    return evaluate_by_attack(
        "GAS-lite", clf, x_test, test_df,
        group_feature_builder=lambda texts: gas_features(
            [char_tokens(t) for t in texts], vocab, risk, idf
        ),
    )


def train_word2vec_vectors(tokenized_train: list[list[str]], tokenized_test: list[list[str]], idf: dict[str, float], name: str):
    from gensim.models import Word2Vec

    print(f"[{name}] training embeddings...")
    start = time.perf_counter()
    w2v = Word2Vec(
        sentences=tokenized_train,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=MIN_COUNT,
        workers=max(os.cpu_count() or 1, 1),
        sg=1,
        seed=RANDOM_STATE,
        epochs=EPOCHS,
    )
    print(f"[{name}] embedding training done in {time.perf_counter() - start:.1f}s")
    return (
        w2v_sentence_vectors(w2v, tokenized_train, idf),
        w2v_sentence_vectors(w2v, tokenized_test, idf),
        w2v,
    )


def run_word2vec_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, object]]:
    train_texts = train_df["text"].tolist()
    test_texts = test_df["text"].tolist()
    y_train = train_df["label"].to_numpy()
    results = []

    print("[Word2Vec-w] tokenizing with jieba...")
    train_word_tokens = [word_tokens(t) for t in train_texts]
    test_word_tokens = [word_tokens(t) for t in test_texts]
    word_idf = build_idf([str(t) for t in train_texts], analyzer=word_tokens)
    x_train_word, x_test_word, word_w2v = train_word2vec_vectors(
        train_word_tokens, test_word_tokens, word_idf, "Word2Vec-w"
    )

    print("[Word2Vec-w+LR] training classifier...")
    start = time.perf_counter()
    word_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    word_lr.fit(x_train_word, y_train)
    print(f"[Word2Vec-w+LR] classifier training done in {time.perf_counter() - start:.1f}s")
    results.extend(evaluate_by_attack(
        "Word2Vec-w+LR", word_lr, x_test_word, test_df,
        group_feature_builder=lambda texts: w2v_sentence_vectors(
            word_w2v, [word_tokens(t) for t in texts], word_idf
        ),
    ))

    print("[Word2Vec-c] tokenizing characters...")
    train_char_tokens = [char_tokens(t) for t in train_texts]
    test_char_tokens = [char_tokens(t) for t in test_texts]
    char_idf = build_idf([str(t) for t in train_texts], analyzer=char_tokens)
    x_train_char, x_test_char, char_w2v = train_word2vec_vectors(
        train_char_tokens, test_char_tokens, char_idf, "Word2Vec-c"
    )

    print("[Word2Vec-c+LR] training classifier...")
    start = time.perf_counter()
    char_lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    char_lr.fit(x_train_char, y_train)
    print(f"[Word2Vec-c+LR] classifier training done in {time.perf_counter() - start:.1f}s")
    char_features = lambda texts: w2v_sentence_vectors(
        char_w2v, [char_tokens(t) for t in texts], char_idf
    )
    results.extend(evaluate_by_attack(
        "Word2Vec-c+LR", char_lr, x_test_char, test_df,
        group_feature_builder=char_features,
    ))

    print("[Word2Vec-c+GBDT] training classifier...")
    start = time.perf_counter()
    gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=150, max_depth=3, learning_rate=0.05)
    gbdt.fit(x_train_char, y_train)
    print(f"[Word2Vec-c+GBDT] classifier training done in {time.perf_counter() - start:.1f}s")
    results.extend(evaluate_by_attack(
        "Word2Vec-c+GBDT", gbdt, x_test_char, test_df,
        group_feature_builder=char_features,
    ))

    return results


def run_doc2vec_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> list[dict[str, object]]:
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument

    print("[Doc2Vec-c] training document embeddings...")
    tagged_docs = [TaggedDocument(words=char_tokens(text), tags=[str(i)]) for i, text in enumerate(train_df["text"].tolist())]
    start = time.perf_counter()
    d2v = Doc2Vec(
        documents=tagged_docs,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=MIN_COUNT,
        workers=max(os.cpu_count() or 1, 1),
        seed=RANDOM_STATE,
        epochs=EPOCHS,
        dm=1,
    )
    print(f"[Doc2Vec-c] embedding training done in {time.perf_counter() - start:.1f}s")

    x_train = np.vstack([d2v.dv[str(i)] for i in range(len(train_df))]).astype(np.float32)
    x_test = np.vstack([d2v.infer_vector(char_tokens(text), epochs=30) for text in test_df["text"].tolist()]).astype(np.float32)
    y_train = train_df["label"].to_numpy()

    print("[Doc2Vec-c+GBDT] training classifier...")
    start = time.perf_counter()
    gbdt = GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=150, max_depth=3, learning_rate=0.05)
    gbdt.fit(x_train, y_train)
    print(f"[Doc2Vec-c+GBDT] classifier training done in {time.perf_counter() - start:.1f}s")

    return evaluate_by_attack(
        "Doc2Vec-c+GBDT", gbdt, x_test, test_df,
        group_feature_builder=lambda texts: np.vstack([
            d2v.infer_vector(char_tokens(text), epochs=30) for text in texts
        ]).astype(np.float32),
    )


def main() -> None:
    train_df, test_df = load_data()
    result_suffix = os.environ.get("RESULT_SUFFIX", "")
    print(f"train samples: {len(train_df)} | test samples: {len(test_df)}")
    print("train labels:", train_df["label"].value_counts().sort_index().to_dict())
    print("test labels:", test_df["label"].value_counts().sort_index().to_dict())

    rows: list[dict[str, object]] = []
    rows.extend(run_gas_baseline(train_df, test_df))
    rows.extend(run_word2vec_baselines(train_df, test_df))
    rows.extend(run_doc2vec_baseline(train_df, test_df))

    result_df = pd.DataFrame(rows)
    out_path = RESULT_DIR / f"classic_baseline_results{result_suffix}.csv"
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    pivot = result_df.pivot(index="attack_type", columns="model", values="f1")
    pivot_path = RESULT_DIR / f"classic_baseline_f1_pivot{result_suffix}.csv"
    pivot.to_csv(pivot_path, encoding="utf-8-sig")

    print("\nF1 by attack type:")
    print(pivot.round(4).fillna(""))
    print(f"\nSaved: {out_path}")
    print(f"Saved: {pivot_path}")


if __name__ == "__main__":
    main()
