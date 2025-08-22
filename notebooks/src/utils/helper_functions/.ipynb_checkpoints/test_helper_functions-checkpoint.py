import numpy as np
from pathlib import Path
import re, numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from tqdm.auto import tqdm
from sklearn.manifold import MDS


def get_sub_paths(path: np.ndarray, n_steps: int, offset: int) -> np.ndarray:
    """
    Gets all sub-paths of length n_steps offset by the offset parameter.

    :param path:        Path to get sub-paths from
    :param n_steps:     Length of each sub-path
    :param offset:      Overlap offset. If 0, paths are all distinct

    :return:            N x n_steps x d tensor of sub-paths, where N is the total number of sub-paths extracted
    """

    total_sub_paths = int(len(path) / (n_steps - offset))

    sub_paths = np.array([
        path[i * (n_steps - offset):i * (n_steps - offset) + n_steps] for i in range(total_sub_paths)
    ])

    return sub_paths


def get_grouped_paths(sub_paths: np.ndarray, n_paths: int) -> np.ndarray:
    """
    Collects sub-paths into equal groups of size n_paths, for calculation using the MMD.

    :param sub_paths:   N x n_steps x d tensor of sub-paths
    :param n_paths:     Number of paths to include in each group
    :return:            M x n_paths x n_steps x d tensor, where M is the total number of groups of size n_paths
                        that can be extracted from the sub_paths object
    """
    total_mmd_paths = int(sub_paths.shape[0] - n_paths + 1)

    return np.array([sub_paths[i:i+n_paths] for i in range(total_mmd_paths)])


def get_memberships(grouped_paths: np.ndarray) -> list:
    """
    Gets the sub-path membership indices given a bank of grouped paths.

    :param grouped_paths:   Bank of paths to compute memberships of
    :return:                Memberships as a ragged list of lists
    """
    shape_vec = grouped_paths.shape
    total_paths = shape_vec[0] + shape_vec[1] - 1

    return [[i for i in range(max(k + 1 - shape_vec[1], 0), min(k + 1, shape_vec[0]))] for k in range(total_paths)]


def get_alphas(membership_vector: list, results: np.ndarray, c_alpha: np.ndarray) -> np.ndarray:
    """
    Gets the plot alpha values of a sub-path given a vector of test results and a memberships list.

    :param membership_vector:   Ragged list of lists corresponding to group memberships
    :param results:             Array of test scores from a given Processor instance
    :param c_alpha:             Critical value associated to Processor instance
    :return:                    Array of alphas for each sub-path corresponding to memberships vector
    """

    mmd_scores = results[1:, :]
    alphas = np.zeros(shape=(results.shape[0]-1, len(membership_vector)))

    for i, m in enumerate(membership_vector):
        if not m:
            continue

        for j, res in enumerate(mmd_scores):
            vec = res[m]
            alphas[j, i] = vec[vec > c_alpha[j]].shape[0] / len(m)

    return alphas

def time_ranges_from_subpaths(subpaths, n_paths: int) -> pd.DataFrame:
    """
    Generate time ranges for each group of subpaths:
    The k-th group interval = [start of the k-th subpath, end of the (k+n_paths-1)-th subpath]
    
    Returns:
        DataFrame with columns: start_time, end_time (as Datetime)
    """
    sp = ensure_regular_3d(subpaths)           # Shape: (N, L, d)
    N = sp.shape[0]
    G = N - n_paths + 1
    if G <= 0:
        raise ValueError(f"Not enough subpaths: N={N}, n_paths={n_paths}")

    starts = [pd.to_datetime(int(sp[k,              0, 0]), unit="s") for k in range(G)]
    ends   = [pd.to_datetime(int(sp[k + n_paths - 1, -1, 0]), unit="s") for k in range(G)]
    return pd.DataFrame({"start_time": starts, "end_time": ends})

def rank_by_size(labels):
    from collections import Counter
    cnt = Counter(labels)
    order = [lab for lab,_ in sorted(cnt.items(), key=lambda x: x[1], reverse=True)]
    remap = {old:new for new,old in enumerate(order)}
    return np.array([remap[l] for l in labels])

def wf_cluster_from_D(dist_matrix: np.ndarray,
                      df_ranges: pd.DataFrame,   # contains start_time / end_time, length = G
                      window_groups: int,
                      n_clusters: int = 3):
    """
    Perform walk-forward clustering using a precomputed group distance matrix (dist_matrix).
    At each time step, only use the historical submatrix D[idxs, idxs] to avoid look-ahead bias.
    
    Returns:
        DataFrame with columns [start_time, end_time, cluster],
        where the first (window_groups - 1) rows have NaN cluster values.
    """
    D = np.asarray(dist_matrix, dtype=float)
    # Defensive step: set diagonal to 0 and enforce symmetry
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)

    G = D.shape[0]
    assert len(df_ranges) == G, "Mismatch: df_ranges length does not match dist_matrix group size"

    labels = [np.nan] * G
    for g in range(window_groups - 1, G):
        idxs = np.arange(g - window_groups + 1, g + 1)  # only use historical window
        Dw = D[np.ix_(idxs, idxs)]
        try:
            model = AgglomerativeClustering(metric="precomputed",
                                            linkage="complete",
                                            n_clusters=n_clusters).fit(Dw)
        except TypeError:  # for backward compatibility with older sklearn
            model = AgglomerativeClustering(affinity="precomputed",
                                            linkage="complete",
                                            n_clusters=n_clusters).fit(Dw)
        labs = rank_by_size(model.labels_)
        labels[g] = int(labs[-1])  # assign label only to the “current group” (last in the window)

    out = df_ranges.copy()
    out["cluster"] = labels
    return out


def daily_regime_from_wf(df_wf, k=3, freq="D"):
    df = df_wf.dropna(subset=["cluster"]).sort_values("end_time")
    if df.empty: 
        return pd.Series(dtype=float)
    ends = pd.to_datetime(df["end_time"]).dt.floor("D")
    labs = df["cluster"].astype(int).to_numpy()
    idx  = pd.date_range(ends.iloc[0], ends.iloc[-1], freq=freq)

    daily = []
    ptr = 0
    for t in idx:
        while ptr < len(ends) and ends.iloc[ptr] <= t:
            ptr += 1
        if ptr == 0: 
            daily.append(np.nan); continue
        s = max(0, ptr - k)
        cand = labs[s:ptr]
        # Majority vote among the most recent k groups
        vals, counts = np.unique(cand, return_counts=True)
        daily.append(vals[np.argmax(counts)])
    return pd.Series(daily, index=idx, name="regime")

def load_and_cluster_all_matrices(
    data_dir: Path,
    trajectory_data,
    window_groups=30,
    n_clusters=3,
    k_vote=3,
    offset=0,
):
    files = sorted(data_dir.glob("pairwise_distance_matrix_steps*_paths*_offset0.npy"))
    results = {}   # key -> dict{series, meta, df_wf}
    pat = re.compile(r"steps(\d+)_paths(\d+)_offset(\d+)\.npy$")

    for f in tqdm(files, desc="Clustering all matrices"):
        m = pat.search(f.name)
        if not m: 
            continue
        n_steps_i, n_paths_i, off_i = map(int, m.groups())
        if off_i != offset: 
            continue

        # Construct the corresponding time ranges for the matrix
        sub_paths_all = get_sub_paths(trajectory_data, n_steps=n_steps_i, offset=off_i)
        df_ranges = time_ranges_from_subpaths(sub_paths_all, n_paths=n_paths_i)
        D = np.load(f)

        # Walk-forward clustering (rolling window)
        df_wf = wf_cluster_from_D(D, df_ranges, window_groups=window_groups, n_clusters=n_clusters)

        # Convert to daily regimes (no look-ahead)
        s_daily = daily_regime_from_wf(df_wf, k=k_vote)

        key = f"s{n_steps_i}_p{n_paths_i}_o{off_i}"
        results[key] = {
            "series": s_daily,
            "meta": {"steps": n_steps_i, "paths": n_paths_i, "offset": off_i,
                     "num_groups": D.shape[0]},
            "df_wf": df_wf,
        }
    return results

def compare_results(results: dict):
    keys = list(results.keys())
    # Find the common daily index (intersection across all series)
    common_idx = None
    for k in keys:
        s = results[k]["series"].dropna()
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    # Summary statistics for each config
    rows = []
    for k in keys:
        s = results[k]["series"].reindex(common_idx).dropna()
        if s.empty: 
            continue
        switches = int((s.diff()!=0).sum())
        dist = s.value_counts(normalize=True).sort_index()
        rows.append({
            "key": k,
            "steps": results[k]["meta"]["steps"],
            "paths": results[k]["meta"]["paths"],
            "cover_days": int(s.notna().sum()),
            "switches": switches,
            "pct_cluster0": dist.get(0, 0.0),
            "pct_cluster1": dist.get(1, 0.0),
            "pct_cluster2": dist.get(2, 0.0),
        })
    summary = pd.DataFrame(rows).sort_values(["steps","paths"]).reset_index(drop=True)

    # Pairwise ARI (Adjusted Rand Index) between configurations
    K = len(keys)
    ari = pd.DataFrame(np.nan, index=keys, columns=keys)
    for i in range(K):
        for j in range(i, K):
            si = results[keys[i]]["series"].reindex(common_idx).dropna()
            sj = results[keys[j]]["series"].reindex(common_idx).dropna()
            idx = si.index.intersection(sj.index)
            if len(idx) == 0: 
                continue
            a = si.loc[idx].astype(int).to_numpy()
            b = sj.loc[idx].astype(int).to_numpy()
            score = adjusted_rand_score(a, b)
            ari.loc[keys[i], keys[j]] = ari.loc[keys[j], keys[i]] = score
    return summary, ari

def ensure_regular_3d(paths) -> np.ndarray:

    arr = np.array(paths, dtype=object)
    if arr.dtype != object:
        return np.asarray(arr, dtype=np.float64, copy=False)

    lens, dims = [], []
    for p in arr:
        p = np.asarray(p)
        if p.ndim != 2:
            raise ValueError(f"Each path must be 2D (time x dim), got {p.shape}")
        lens.append(p.shape[0]); dims.append(p.shape[1])
    if len(set(dims)) != 1:
        raise ValueError(f"Inconsistent path dimension: {set(dims)}")

    L, d = int(min(lens)), int(dims[0])
    out = np.empty((len(arr), L, d), dtype=np.float64)
    for i, p in enumerate(arr):
        out[i] = np.asarray(p, dtype=np.float64)[:L]
    return out


from sklearn.cluster import AgglomerativeClustering

def _group_forward_returns(df_ranges, px: pd.Series, H: int, start_from: str = "next") -> np.ndarray:
    """
    Compute forward log returns for each group after its end_time.

    Parameters
    ----------
    df_ranges : pd.DataFrame
        Must contain column 'end_time'.
    px : pd.Series
        Price series with datetime index.
    H : int
        Forward horizon (number of steps).
    start_from : {"next", "same"}
        "next" = start counting returns from the bar AFTER end_time.
        "same" = start counting from end_time itself (if aligned with index).

    Returns
    -------
    np.ndarray
        Forward log returns of length = len(df_ranges).
    """
    px = px if isinstance(px, pd.Series) else px.iloc[:, 0]
    lp = np.log(px.dropna())
    idx = lp.index
    end = pd.to_datetime(df_ranges["end_time"])

    if start_from == "same":
        i0 = idx.searchsorted(end, side="left")   # same bar (if aligned)
    else:  # "next"
        i0 = idx.searchsorted(end, side="right")  # strictly after end_time

    iH = i0 + H  # strictly forward H steps
    r = np.full(len(df_ranges), np.nan, dtype=float)
    ok = (iH < len(idx))
    r[ok] = (lp.values[iH[ok]] - lp.values[i0[ok]])
    return r


def fit_prototypes_and_mapping(
    D, df_ranges, price_series, *, train_end, n_clusters=3,
    H=8, min_count=5, tau_pos=0.0, tau_neg=0.0, use_median=False, start_from="next"
):
    """
    Training phase: cluster → select medoids → assign semantic labels (0=Bear, 1=Neutral, 2=Bull)
    based on forward returns in the training window.

    Rules:
    1) Rank clusters by average/median forward return (worst → best).
       Lowest = candidate bear, highest = candidate bull (relative only).
    2) Only assign bull if best_mean > tau_pos; only assign bear if worst_mean < tau_neg;
       otherwise keep as neutral.
    3) No forced fallback to guarantee both bull/bear. This avoids mislabeling
       in strongly trending markets.
    """
    # -- clean distance matrix --
    D = np.asarray(D, dtype=float)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)

    # -- forward returns for groups within training window --
    rH = _group_forward_returns(df_ranges, price_series, H, start_from=start_from)
    train_mask = pd.to_datetime(df_ranges["end_time"]) <= pd.to_datetime(train_end)
    idx_train  = np.where(train_mask)[0]
    if len(idx_train) < n_clusters:
        raise ValueError("Not enough training samples to cluster into n_clusters.")

    # -- clustering on training subset --
    Dt = D[np.ix_(idx_train, idx_train)]
    try:
        mdl = AgglomerativeClustering(metric="precomputed", linkage="complete",
                                      n_clusters=n_clusters).fit(Dt)
    except TypeError:
        mdl = AgglomerativeClustering(affinity="precomputed", linkage="complete",
                                      n_clusters=n_clusters).fit(Dt)
    labs_t = mdl.labels_

    # -- compute medoid for each cluster (row with smallest sum of distances) --
    medoid_idx = []
    for c in range(n_clusters):
        members = idx_train[labs_t == c]
        if len(members) == 0:
            medoid_idx.append(None); continue
        sub = D[np.ix_(members, members)]
        medoid_idx.append(int(members[np.argmin(sub.sum(axis=1))]))

    # -- cluster-level return stats (mean/median + sample count) --
    effect = {}
    counts = {}
    for c in range(n_clusters):
        vals = rH[idx_train[labs_t == c]]
        vals = vals[np.isfinite(vals)]
        counts[c] = int(len(vals))
        if counts[c] > 0:
            effect[c] = float(np.median(vals) if use_median else np.mean(vals))

    # keep only clusters with enough samples
    eligible = {c: m for c, m in effect.items() if counts.get(c, 0) >= min_count}

    # default: all neutral
    mapping = {c: 1 for c in range(n_clusters)}

    if eligible:
        # candidate bear/bull by sorted effect
        order = sorted(eligible.items(), key=lambda x: x[1])
        worst_c, worst_mean = order[0]
        best_c,  best_mean  = order[-1]

        # thresholds = strength check
        if best_mean  > tau_pos:
            mapping[best_c] = 2   # bull
        if worst_mean < tau_neg:
            mapping[worst_c] = 0  # bear

    # -- reverse lookup: prototype index → cluster id --
    proto2cluster = {medoid_idx[c]: c for c in range(n_clusters) if medoid_idx[c] is not None}

    # -- debugging printout --
    debug_df = (pd.DataFrame({
        "cluster": list(range(n_clusters)),
        "count":   [counts.get(c, 0) for c in range(n_clusters)],
        "effect":  [effect.get(c, np.nan) for c in range(n_clusters)],
        "label":   [mapping.get(c, 1) for c in range(n_clusters)],
    }).sort_values("effect"))
    print("=== Train cluster stats (sorted by effect) ===")
    print(debug_df.to_string(index=False))

    return medoid_idx, proto2cluster, mapping


def assign_by_prototypes(D, medoid_idx, proto2cluster, mapping):
    """
    Assign clusters over the full period:
    1) For each group, find nearest prototype (medoid).
    2) Map prototype → cluster_id → semantic label (0/1/2).
    """
    D = np.asarray(D, dtype=float)
    meds = [i for i in medoid_idx if i is not None]
    if not meds:
        return np.full(D.shape[0], np.nan), np.full(D.shape[0], np.nan)
    d_to_med = D[:, meds]                      # (groups × prototypes)
    nearest  = [meds[int(i)] for i in np.argmin(d_to_med, axis=1)]
    cluster_id = [proto2cluster.get(j, np.nan) for j in nearest]
    dir_label  = [mapping.get(c, 1) if np.isfinite(c) else np.nan for c in cluster_id]
    return np.array(cluster_id, dtype=float), np.array(dir_label, dtype=float)


def compress_with_rolling_vote(df_wf, k: int = 3, half_life_groups: float | None = None, freq: str = "D"):
    """
    Rolling vote regime labeling:
    - For each day, look back at the last k completed groups (end_time <= t).
    - Use majority vote (or exponential decay weights if half_life_groups is set).
    - Generate daily regime labels and compress into non-overlapping spans.
    - Purely backward-looking → no lookahead leakage.
    """
    df = df_wf.dropna(subset=["cluster"]).copy().sort_values("end_time")
    if df.empty:
        return pd.DataFrame(columns=["start","end","cluster"])

    # ensure timestamps are aligned to daily
    ends = pd.to_datetime(df["end_time"]).dt.floor("D")
    labs = df["cluster"].astype(int).to_numpy()

    # daily index from first to last labeled group
    idx = pd.date_range(ends.iloc[0], ends.iloc[-1], freq=freq)

    daily_label = []
    ptr = 0  # number of finished groups (end_time <= t)
    for t in idx:
        while ptr < len(ends) and ends.iloc[ptr] <= t:
            ptr += 1
        if ptr == 0:
            daily_label.append(np.nan)
            continue

        s = max(0, ptr - k)
        cand = labs[s:ptr]

        if half_life_groups is None:
            vals, counts = np.unique(cand, return_counts=True)
            vote_winner = vals[np.argmax(counts)]
        else:
            n = cand.shape[0]
            ages = np.arange(n-1, -1, -1)   # 0 = most recent
            lam = np.log(2) / max(half_life_groups, 1e-6)
            w = np.exp(-lam * ages)
            scores = {u: w[(cand == u)].sum() for u in np.unique(cand)}
            vote_winner = max(scores.items(), key=lambda x: x[1])[0]

        daily_label.append(vote_winner)

    # merge consecutive spans with same label
    s = pd.Series(daily_label, index=idx).dropna()
    spans = []
    cur = None
    for t, lab in s.items():
        if lab != cur:
            if cur is not None:
                spans.append((start, prev, cur))
            cur = lab
            start = t
        prev = t
    spans.append((start, prev, cur))
    return pd.DataFrame(spans, columns=["start","end","cluster"])


def plot_price_with_cluster_background(price_series, df_clusters, title="", label_col="cluster", label_names=None):
    """
    Plot price series with colored background spans for cluster/regime labels.
    """
    import matplotlib.pyplot as plt, matplotlib.dates as mdates, seaborn as sns
    if hasattr(price_series, "columns"): price_series = price_series.iloc[:,0]
    df = df_clusters.copy()

    # align column names
    if {"start","end"}.issubset(df.columns):
        df = df.rename(columns={"start":"start_time","end":"end_time"})

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(price_series.index, price_series.values, color="black", lw=1.5, label=price_series.name or "Price")

    df = df.dropna(subset=[label_col])
    uniq = sorted(df[label_col].unique())
    cmap = sns.color_palette("Set2", n_colors=len(uniq))
    color_map = {cl: cmap[i] for i, cl in enumerate(uniq)}

    for _, row in df.iterrows():
        ax.axvspan(row["start_time"], row["end_time"],
                   color=color_map[row[label_col]], alpha=0.25,
                   label=(label_names.get(int(row[label_col]), f"{label_col} {int(row[label_col])}")
                          if label_names else f"{label_col} {int(row[label_col])}"))

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left")

    ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.grid(True)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.tight_layout(); plt.show()
    fig.savefig("/Users/zheguan/DDA/Signature/Bit/newrepo/Bit_regime_match/notebooks/dataall/wf_outputs/images/" f"{title}.png", dpi=200, bbox_inches="tight")

# —— Convert spans into daily frequency signals (for backtesting), then save as CSV (index = date) ——
def spans_to_daily(spans: pd.DataFrame, freq: str = "D") -> pd.Series:
    # spans: columns = ["start","end","cluster"]
    if spans.empty:
        return pd.Series(dtype=float)
    chunks = []
    for _, row in spans.iterrows():
        idx = pd.date_range(pd.to_datetime(row["start"]), pd.to_datetime(row["end"]), freq=freq)
        chunks.append(pd.Series(int(row["cluster"]), index=idx))
    s = pd.concat(chunks).sort_index()
    # If there are overlapping dates, keep the last occurrence
    return s[~s.index.duplicated(keep="last")].rename("dir_label")

def load_D(path):
    """Load a distance matrix from .npy file, clean & symmetrize."""
    D = np.load(path)
    D = np.asarray(D, float)
    np.fill_diagonal(D, 0.0)                        # set diagonal = 0
    D = 0.5 * (D + D.T)                             # symmetrize
    D[~np.isfinite(D)] = np.nanmax(D[np.isfinite(D)])  # replace NaN/inf
    D[D < 0] = 0.0
    return D

def agglom_precomputed(D, n_clusters=3):
    """Perform Agglomerative Clustering with precomputed distance matrix."""
    try:
        model = AgglomerativeClustering(metric="precomputed",
                                        linkage="complete",
                                        n_clusters=n_clusters).fit(D)
    except TypeError:  # backward compatibility for older sklearn
        model = AgglomerativeClustering(affinity="precomputed",
                                        linkage="complete",
                                        n_clusters=n_clusters).fit(D)
    return model.labels_

def mds_embed(D, random_state=42):
    """Apply MDS to embed distance matrix into 2D coordinates."""
    try:
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=random_state, n_init=4, max_iter=300,
                  normalized_stress="auto")
    except TypeError:  # fallback for older sklearn
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=random_state, n_init=4, max_iter=300)
    return mds.fit_transform(D)

def plot_and_save(D, labels, save_path, title="Trajectory Clusters (MDS Projection)"):
    """Plot clusters in MDS 2D space and save to a file."""
    emb = mds_embed(D)
    plt.figure(figsize=(8, 6))
    for c in np.unique(labels.astype(int)):
        idx = (labels == c)
        plt.scatter(emb[idx, 0], emb[idx, 1], s=18, label=f"Cluster {c+1}")
    plt.title(title)
    plt.xlabel("MDS-1"); plt.ylabel("MDS-2")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)   # save as PNG
    plt.close()
    print(f"✓ Saved figure: {save_path}")




