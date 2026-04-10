import numpy as np


def _topk_count(n, k_ratio):
    return max(1, int(n * k_ratio))


def precision_at_k(y, scores, k_ratio=0.1):
    n = _topk_count(len(y), k_ratio)
    idx = np.argsort(scores)[::-1][:n]
    return y[idx].mean()


def recall_at_k(y, scores, k_ratio=0.1):
    n = _topk_count(len(y), k_ratio)
    idx = np.argsort(scores)[::-1][:n]
    return y[idx].sum() / (y.sum() + 1e-8)


def ndcg_at_k(y, scores, k_ratio=0.1):
    n = _topk_count(len(y), k_ratio)
    idx = np.argsort(scores)[::-1][:n]
    gains = y[idx]
    discounts = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_idx = np.argsort(y)[::-1][:n]
    ideal_gains = y[ideal_idx]
    idcg = float(np.sum(ideal_gains * discounts))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(y, scores, k_ratio=0.1):
    n = _topk_count(len(y), k_ratio)
    idx = np.argsort(scores)[::-1][:n]
    return float(np.any(y[idx] > 0))


def _pearson_corr(x, z):
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    x_centered = x - x.mean()
    z_centered = z - z.mean()
    denom = np.sqrt(np.sum(x_centered ** 2) * np.sum(z_centered ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(x_centered * z_centered) / denom)


def _average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = avg_rank
        start = end

    return ranks


class _FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def update(self, index, delta=1):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        total = 0
        while index > 0:
            total += self.tree[index]
            index -= index & -index
        return total


def _count_tied_pairs(values):
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts * (counts - 1) // 2))


def _kendall_tau_b(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 2:
        return 0.0

    order = np.lexsort((y, x))
    x_sorted = x[order]
    y_sorted = y[order]

    _, y_inverse = np.unique(y_sorted, return_inverse=True)
    y_ranks = y_inverse + 1
    tree = _FenwickTree(int(y_ranks.max()))

    concordant = 0
    discordant = 0
    total_seen = 0
    start = 0

    while start < n:
        end = start + 1
        while end < n and x_sorted[end] == x_sorted[start]:
            end += 1

        for yi in y_ranks[start:end]:
            less = tree.query(yi - 1)
            greater = total_seen - tree.query(yi)
            concordant += less
            discordant += greater

        for yi in y_ranks[start:end]:
            tree.update(yi, 1)
            total_seen += 1

        start = end

    total_pairs = n * (n - 1) // 2
    ties_x = _count_tied_pairs(x)
    ties_y = _count_tied_pairs(y)
    denom = np.sqrt((total_pairs - ties_x) * (total_pairs - ties_y))
    if denom <= 0:
        return 0.0
    return float((concordant - discordant) / denom)


def spearman_corr(y, scores):
    return _pearson_corr(_average_ranks(y), _average_ranks(scores))


def kendall_tau(y, scores):
    return _kendall_tau_b(y, scores)


def evaluate_ranking(y, scores):
    return {
        "p10": precision_at_k(y, scores, 0.1),
        "r10": recall_at_k(y, scores, 0.1),
        "p20": precision_at_k(y, scores, 0.2),
        "r20": recall_at_k(y, scores, 0.2),
        "ndcg10": ndcg_at_k(y, scores, 0.1),
        "ndcg20": ndcg_at_k(y, scores, 0.2),
        "hit10": hit_rate_at_k(y, scores, 0.1),
        "hit20": hit_rate_at_k(y, scores, 0.2),
        "spearman": spearman_corr(y, scores),
        "kendall_tau": kendall_tau(y, scores),
    }
