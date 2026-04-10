# feature_blocks.py
import json
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
from collections import defaultdict, deque
from datetime import timedelta


def build_topic_features(records, train_idx):
    train_topics = [records[i]["topicCategories"] for i in train_idx]
    all_topics   = [r["topicCategories"] for r in records]

    mlb = MultiLabelBinarizer(sparse_output=False)
    mlb.fit(train_topics)
    X_topic = mlb.transform(all_topics).astype(np.float32)

    return X_topic, mlb


def build_topic_monthly_features(records, train_idx, val_idx, test_idx, alpha=5.0):
    monthly_features = np.zeros((len(records), 3), dtype=np.float32)
    month_total_count = defaultdict(int)
    month_topic_count = defaultdict(int)
    global_topic_count = defaultdict(int)
    global_total_count = 0

    sorted_indices = sorted(
        range(len(records)),
        key=lambda idx: (records[idx]["date"], idx),
    )

    for idx in sorted_indices:
        record = records[idx]
        month_key = (record["date"].year, record["date"].month)
        topics = [
            topic
            for topic in set(record.get("topicCategories", []))
            if topic
        ]
        if not topics:
            continue

        month_count = month_total_count.get(month_key, 0)
        month_topic_probs = []
        topic_global_probs = []

        for topic in topics:
            global_prob = (
                global_topic_count.get(topic, 0) / global_total_count
                if global_total_count > 0 else 0.0
            )
            topic_global_probs.append(global_prob)

            if month_count > 0:
                topic_count = month_topic_count.get((month_key, topic), 0)
                month_prob = (topic_count + alpha * global_prob) / (month_count + alpha)
            else:
                month_prob = global_prob
            month_topic_probs.append(month_prob)

        mean_month_prob = float(np.mean(month_topic_probs))
        max_month_prob = float(np.max(month_topic_probs))
        mean_global_prob = float(np.mean(topic_global_probs))

        monthly_features[idx, 0] = mean_month_prob
        monthly_features[idx, 1] = max_month_prob
        monthly_features[idx, 2] = mean_month_prob - mean_global_prob

        month_total_count[month_key] += 1
        global_total_count += 1
        for topic in topics:
            month_topic_count[(month_key, topic)] += 1
            global_topic_count[topic] += 1

    scaler = StandardScaler()
    monthly_features[train_idx] = scaler.fit_transform(monthly_features[train_idx])
    monthly_features[val_idx] = scaler.transform(monthly_features[val_idx])
    monthly_features[test_idx] = scaler.transform(monthly_features[test_idx])

    return monthly_features


def build_channel_context_features(
    records,
    channels_array,
    train_idx,
    val_idx,
    test_idx,
    recent_days=30,
):
    features = np.zeros((len(records), 2), dtype=np.float32)
    sorted_indices = sorted(
        range(len(records)),
        key=lambda idx: (records[idx]["date"], idx),
    )
    history_by_channel = defaultdict(deque)
    cumulative_sum_by_channel = defaultdict(float)
    cumulative_count_by_channel = defaultdict(int)
    global_sum = 0.0
    global_count = 0

    for idx in sorted_indices:
        record = records[idx]
        channel = channels_array[idx]
        date_value = record["date"]
        comment_rate = float(record["comment_rate"])

        history = history_by_channel[channel]
        cutoff = date_value - timedelta(days=recent_days)
        while history and history[0][0] < cutoff:
            history.popleft()

        prior_count = cumulative_count_by_channel[channel]
        prior_mean = (
            cumulative_sum_by_channel[channel] / prior_count
            if prior_count > 0 else (
                global_sum / global_count if global_count > 0 else 0.0
            )
        )
        recent_mean = (
            float(np.mean([value for _, value in history]))
            if history else prior_mean
        )

        features[idx, 0] = float(np.log1p(prior_count))
        features[idx, 1] = recent_mean

        history.append((date_value, comment_rate))
        cumulative_sum_by_channel[channel] += comment_rate
        cumulative_count_by_channel[channel] += 1
        global_sum += comment_rate
        global_count += 1

    scaler = StandardScaler()
    features[train_idx] = scaler.fit_transform(features[train_idx])
    features[val_idx] = scaler.transform(features[val_idx])
    features[test_idx] = scaler.transform(features[test_idx])
    return features


def build_channel_metadata_features(
    records,
    train_idx,
    val_idx,
    test_idx,
    metadata_path="data/channels_full_metadata.json",
):
    with open(metadata_path, "r", encoding="utf-8") as f:
        channels_meta = json.load(f)

    def _safe_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return 0

    channel_features = {}
    for channel_id, payload in channels_meta.items():
        stats = payload.get("statistics", {})
        snippet = payload.get("snippet", {})
        topic_details = payload.get("topicDetails", {})

        subs = _safe_int(stats.get("subscriberCount", 0))
        views = _safe_int(stats.get("viewCount", 0))
        videos = _safe_int(stats.get("videoCount", 0))
        desc_len = len(snippet.get("description", "") or "")
        has_custom = float(bool(snippet.get("customUrl")))
        country_tw = float((snippet.get("country") or "").upper() == "TW")
        topic_count = float(len(topic_details.get("topicCategories", []) or []))

        channel_features[channel_id] = np.array(
            [
                np.log1p(subs),
                np.log1p(views),
                np.log1p(videos),
                np.log1p(desc_len),
                has_custom,
                country_tw,
                topic_count,
            ],
            dtype=np.float32,
        )

    fallback = np.mean(np.stack(list(channel_features.values())), axis=0)
    out = np.zeros((len(records), fallback.shape[0]), dtype=np.float32)

    for i, record in enumerate(records):
        channel_id = record.get("channel_id", "")
        out[i] = channel_features.get(channel_id, fallback)

    scaler = StandardScaler()
    out[train_idx] = scaler.fit_transform(out[train_idx])
    out[val_idx] = scaler.transform(out[val_idx])
    out[test_idx] = scaler.transform(out[test_idx])
    return out


def build_embeddings(records, model_name, prefix_mode="none"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(model_name, device=device)

    texts = [x["framing_text"] for x in records]

    # ---- Prefix control ----
    if prefix_mode == "e5":
        texts = ["passage: " + t for t in texts]
    elif prefix_mode == "none":
        pass
    else:
        raise ValueError("prefix_mode must be 'none' or 'e5'")


    batch_size = 256

    E_framing = embedder.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True
    ).astype(np.float32)

    return E_framing, embedder

def build_duration_feature(records, train_idx, val_idx, test_idx):

    durations = np.array([
        r.get("duration_seconds", 0)
        for r in records
    ], dtype=np.float32).reshape(-1,1)

    durations = np.log1p(durations)

    scaler = StandardScaler()

    durations[train_idx] = scaler.fit_transform(durations[train_idx])
    durations[val_idx] = scaler.transform(durations[val_idx])
    durations[test_idx] = scaler.transform(durations[test_idx])

    return durations


def build_object_detection_features(
    records,
    train_idx,
    val_idx,
    test_idx,
    feature_path="experiments/cache/yolov8_detection_features.csv",
):
    feature_df = pd.read_csv(feature_path)
    if "video_id" not in feature_df.columns:
        raise ValueError("YOLO feature CSV must include a `video_id` column.")

    video_ids = [record["video_id"] for record in records]
    feature_df = feature_df.set_index("video_id")

    numeric_cols = [
        col
        for col in feature_df.columns
        if col not in {"channel_slug"}
    ]
    aligned = feature_df.reindex(video_ids)
    aligned_numeric = aligned[numeric_cols].fillna(0.0).astype(np.float32)

    x_obj = aligned_numeric.to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_obj[train_idx] = scaler.fit_transform(x_obj[train_idx])
    x_obj[val_idx] = scaler.transform(x_obj[val_idx])
    x_obj[test_idx] = scaler.transform(x_obj[test_idx])

    return x_obj, numeric_cols
