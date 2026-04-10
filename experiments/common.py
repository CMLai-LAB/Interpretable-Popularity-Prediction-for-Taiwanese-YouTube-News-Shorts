import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_blocks import (
    build_channel_context_features,
    build_channel_metadata_features,
    build_duration_feature,
    build_embeddings,
    build_object_detection_features,
    build_topic_monthly_features,
    build_topic_features,
)
from labels import build_channel_normalized_labels
from load_data import load_data


DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FRAMING_FEATURES = [
    "title_len",
    "exclaim",
    "question",
    "digit_ratio",
    "ocr_len",
    "ocr_exclaim",
    "ocr_question",
    "ocr_digit_ratio",
]
DURATION_FEATURE_NAMES = ["log_duration_seconds"]
TOPIC_MONTHLY_FEATURE_NAMES = [
    "topic_month_mean_prob",
    "topic_month_max_prob",
    "topic_month_mean_delta_vs_global",
]
CHANNEL_FEATURE_NAMES = [
    "channel_size_log",
    "channel_recent_performance",
    "channel_log_subscribers",
    "channel_log_total_views",
    "channel_log_video_count",
    "channel_log_description_len",
    "channel_has_custom_url",
    "channel_country_tw",
    "channel_topic_count",
]
NUMERIC_FEATURE_NAMES = FRAMING_FEATURES + DURATION_FEATURE_NAMES
THEORY_GROUP_LABELS = {
    "SEM": "semantic_content",
    "FRM": "framing_cues",
    "CTX": "contextual_channel_cues",
}


def stage(msg):
    print(f"\n{'=' * 60}")
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    print(f"{'=' * 60}")


def build_split_indices(records):
    train_idx = []
    val_idx = []
    test_idx = []

    for i, record in enumerate(records):
        date_value = record["date"]

        if date_value.year == 2025 and date_value.month <= 10:
            train_idx.append(i)
        elif date_value.year == 2025 and date_value.month >= 11:
            val_idx.append(i)
        elif date_value.year == 2026:
            test_idx.append(i)

    return (
        np.array(train_idx),
        np.array(val_idx),
        np.array(test_idx),
    )


def build_framing_matrix(records, train_idx, val_idx, test_idx):
    df = pd.DataFrame(records)
    framing_matrix = df[FRAMING_FEATURES].values.astype(np.float32)
    duration_matrix = build_duration_feature(records, train_idx, val_idx, test_idx)

    scaler = StandardScaler()
    framing_matrix[train_idx] = scaler.fit_transform(framing_matrix[train_idx])
    framing_matrix[val_idx] = scaler.transform(framing_matrix[val_idx])
    framing_matrix[test_idx] = scaler.transform(framing_matrix[test_idx])

    return np.concatenate([framing_matrix, duration_matrix], axis=1)


def build_feature_names(x_embed, mlb, object_feature_names=None):
    embed_feature_names = [f"embed_{i}" for i in range(x_embed.shape[1])]
    topic_feature_names = [f"topic_{topic}" for topic in mlb.classes_]
    object_feature_names = object_feature_names or []

    return (
        embed_feature_names
        + topic_feature_names
        + CHANNEL_FEATURE_NAMES
        + TOPIC_MONTHLY_FEATURE_NAMES
        + NUMERIC_FEATURE_NAMES
        + object_feature_names
    )


def build_feature_groups(x_embed, mlb, object_feature_names=None):
    groups = []
    next_idx = 0
    object_feature_names = object_feature_names or []

    embed_dim = x_embed.shape[1]
    groups.append(
        {
            "group_name": "embedding",
            "feature_names": [f"embed_{i}" for i in range(embed_dim)],
            "indices": list(range(next_idx, next_idx + embed_dim)),
        }
    )
    next_idx += embed_dim

    for topic in mlb.classes_:
        groups.append(
            {
                "group_name": f"topic_{topic}",
                "feature_names": [f"topic_{topic}"],
                "indices": [next_idx],
            }
        )
        next_idx += 1

    for feature_name in CHANNEL_FEATURE_NAMES:
        groups.append(
            {
                "group_name": feature_name,
                "feature_names": [feature_name],
                "indices": [next_idx],
            }
        )
        next_idx += 1

    for feature_name in TOPIC_MONTHLY_FEATURE_NAMES:
        groups.append(
            {
                "group_name": feature_name,
                "feature_names": [feature_name],
                "indices": [next_idx],
            }
        )
        next_idx += 1

    for feature_name in NUMERIC_FEATURE_NAMES:
        groups.append(
            {
                "group_name": feature_name,
                "feature_names": [feature_name],
                "indices": [next_idx],
            }
        )
        next_idx += 1

    for feature_name in object_feature_names:
        groups.append(
            {
                "group_name": feature_name,
                "feature_names": [feature_name],
                "indices": [next_idx],
            }
        )
        next_idx += 1

    return groups


def split_xy(x, y, train_idx, val_idx, test_idx):
    return {
        "X_train": x[train_idx],
        "X_val": x[val_idx],
        "X_test": x[test_idx],
        "y_train": y[train_idx],
        "y_val": y[val_idx],
        "y_test": y[test_idx],
    }


def build_feature_blocks(context):
    channel_context = context["channel_context_features"]
    framing_matrix = np.concatenate(
        [context["X_topic_monthly"], context["X_framing"]],
        axis=1,
    )
    semantic_matrix = np.concatenate(
        [context["X_embed"], context["X_topiccat"]],
        axis=1,
    )
    blocks = {
        "E": context["X_embed"],
        "T": context["X_topiccat"],
        "C_CTX": channel_context,
        "C_META": context["channel_metadata_features"],
        "C": context["channel_feature_matrix"],
        "M": context["X_topic_monthly"],
        "F": context["X_framing"],
        "SEM": semantic_matrix,
        "FRM": framing_matrix,
        "CTX": context["channel_feature_matrix"],
    }
    if "X_object" in context:
        blocks["V_OBJ"] = context["X_object"]
    return blocks


def concatenate_feature_blocks(blocks, keys):
    return np.concatenate([blocks[key] for key in keys], axis=1).astype(np.float32)


def _rebuild_full_matrix(context):
    matrix_parts = [
        context["X_embed"],
        context["X_topiccat"],
        context["channel_feature_matrix"],
        context["X_topic_monthly"],
        context["X_framing"],
    ]
    object_feature_names = context.get("object_feature_names", [])
    if "X_object" in context:
        matrix_parts.append(context["X_object"])

    x_all = np.concatenate(matrix_parts, axis=1)

    context["X_all"] = x_all
    context["feature_names"] = build_feature_names(
        context["X_embed"],
        context["mlb"],
        object_feature_names=object_feature_names,
    )
    context["feature_groups"] = build_feature_groups(
        context["X_embed"],
        context["mlb"],
        object_feature_names=object_feature_names,
    )

    assert len(context["feature_names"]) == x_all.shape[1], "Feature name mismatch!"
    return context


def apply_label_configuration(
    base_context,
    label_percentile=80,
    label_train_idx=None,
    verbose=True,
):
    context = dict(base_context)
    if label_train_idx is None:
        label_train_idx = context["train_idx"]
    label_train_idx = np.asarray(label_train_idx)

    y, channels_array = build_channel_normalized_labels(
        context["records"],
        label_train_idx,
        percentile=label_percentile,
    )
    channel_feature_matrix = np.concatenate(
        [
            context["channel_context_features"],
            context["channel_metadata_features"],
        ],
        axis=1,
    )

    context.update(
        {
            "y": y,
            "channels_array": channels_array,
            "channel_feature_matrix": channel_feature_matrix,
            "label_percentile": label_percentile,
            "label_train_idx": label_train_idx,
        }
    )

    if "X_embed" in context:
        context = _rebuild_full_matrix(context)

    if verbose:
        print(f"Label percentile: {label_percentile}")
        print("Channel feature matrix shape:", channel_feature_matrix.shape)

    return context


def build_base_context(
    records=None,
    train_idx=None,
    val_idx=None,
    test_idx=None,
    label_percentile=80,
    label_train_idx=None,
    object_feature_path=None,
    verbose=True,
):
    if records is None:
        if verbose:
            stage("Loading JSON files")
        records = load_data()

    if train_idx is None or val_idx is None or test_idx is None:
        train_idx, val_idx, test_idx = build_split_indices(records)
    else:
        train_idx = np.asarray(train_idx)
        val_idx = np.asarray(val_idx)
        test_idx = np.asarray(test_idx)

    x_framing = build_framing_matrix(records, train_idx, val_idx, test_idx)

    if verbose:
        print(len(train_idx), len(val_idx), len(test_idx))
        channel_counts = pd.Series([records[i]["channel"] for i in val_idx]).value_counts()
        print("\nChannel distribution:")
        print(channel_counts)

    x_topiccat, mlb = build_topic_features(records, train_idx)
    x_topic_monthly = build_topic_monthly_features(records, train_idx, val_idx, test_idx)
    channels_array = np.array([r["channel"] for r in records])
    channel_context_features = build_channel_context_features(
        records,
        channels_array,
        train_idx,
        val_idx,
        test_idx,
    )
    channel_metadata_features = build_channel_metadata_features(
        records,
        train_idx,
        val_idx,
        test_idx,
    )
    x_object = None
    object_feature_names = []
    if object_feature_path is not None:
        x_object, object_feature_names = build_object_detection_features(
            records,
            train_idx,
            val_idx,
            test_idx,
            feature_path=object_feature_path,
        )

    context = {
        "records": records,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "X_topiccat": x_topiccat,
        "X_topic_monthly": x_topic_monthly,
        "X_framing": x_framing,
        "channel_context_features": channel_context_features,
        "channel_metadata_features": channel_metadata_features,
        "mlb": mlb,
        "object_feature_path": object_feature_path,
    }
    if x_object is not None:
        context["X_object"] = x_object
        context["object_feature_names"] = object_feature_names
    context = apply_label_configuration(
        context,
        label_percentile=label_percentile,
        label_train_idx=label_train_idx,
        verbose=verbose,
    )

    if verbose:
        print("Channel context feature shape:", channel_context_features.shape)
        print("Channel metadata feature shape:", channel_metadata_features.shape)
        print("Topic monthly feature shape:", x_topic_monthly.shape)
        if x_object is not None:
            print("Object feature shape:", x_object.shape)

    return context


def attach_embedding_features(
    base_context,
    embedding_model,
    prefix_mode="none",
    x_embed=None,
    embedder=None,
    embedding_time_sec=None,
    verbose=True,
):
    context = dict(base_context)

    if x_embed is None:
        if verbose:
            stage("Generating embeddings")
        embedding_start_time = time.time()
        x_embed, embedder = build_embeddings(
            context["records"],
            embedding_model,
            prefix_mode=prefix_mode,
        )
        embedding_time_sec = time.time() - embedding_start_time
    elif embedding_time_sec is None:
        embedding_time_sec = 0.0

    context.update(
        {
            "X_embed": x_embed,
            "embedder": embedder,
            "embedding_model": embedding_model,
            "embedding_time_sec": embedding_time_sec,
        }
    )
    context = _rebuild_full_matrix(context)

    if verbose:
        print(f"  Embedding: {x_embed.shape[1]} dims")
        print(f"  Topic categories: {context['X_topiccat'].shape[1]} dims")
        print(f"  Channel features: {context['channel_feature_matrix'].shape[1]} dims")
        print(f"  Topic monthly: {context['X_topic_monthly'].shape[1]} dims")
        print(f"  Numeric features: {len(NUMERIC_FEATURE_NAMES)} dims")
        print(f"  Total: {context['X_all'].shape[1]} dims")
        print("Total feature names:", len(context["feature_names"]))
        print("Total feature dim :", context["X_all"].shape[1])

    return context


def build_experiment_context(
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    prefix_mode="none",
    label_percentile=80,
    object_feature_path=None,
):
    base_context = build_base_context(
        label_percentile=label_percentile,
        object_feature_path=object_feature_path,
    )
    return attach_embedding_features(
        base_context,
        embedding_model=embedding_model,
        prefix_mode=prefix_mode,
    )
