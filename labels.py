# labels.py
import numpy as np

def build_channel_normalized_labels(records, train_idx, percentile=80):
    rates = np.array([x["comment_rate"] for x in records])
    channels_array = np.array([r["channel"] for r in records])

    y = np.zeros(len(records), dtype=int)
    unique_channels = np.unique(channels_array)

    for ch in unique_channels:
        ch_train_mask = (channels_array == ch) & np.isin(
            np.arange(len(records)), train_idx
        )

        if ch_train_mask.sum() == 0:
            continue

        ch_threshold = np.percentile(rates[ch_train_mask], percentile)

        ch_all_mask = (channels_array == ch)
        y[ch_all_mask] = (rates[ch_all_mask] >= ch_threshold).astype(int)

    return y, channels_array