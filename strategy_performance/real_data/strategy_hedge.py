def backtest_lead_lag_hedge(
    prices: pd.DataFrame,
    lead_lag_matrix_rolling: pd.DataFrame,
    quantile: float = 0.1,
    invert: bool = False,          # if True, swap long/short legs
    min_names: int = 1,            # minimum names per leg; skip day if not satisfied
    use_log_returns: bool = True,  # True: log returns; False: simple returns
) -> pd.Series:
    """
    Lead–lag hedged strategy: by default go long followers and short leaders.
    If `invert=True`, reverse the legs.

    Parameters
    ----------
    prices : pd.DataFrame
        Price table indexed by trading dates; columns are assets.
    lead_lag_matrix_rolling : pd.DataFrame
        Time series of rolling lead–lag matrices; index = dates. Each entry at date d
        should be a square matrix (DataFrame) aligned with `prices.columns`.
    quantile : float
        Tail percentile for selecting leaders (positive tail) and followers (negative tail).
    invert : bool
        If True, swap leaders and followers (i.e., long leaders / short followers).
    min_names : int
        Minimum number of names required in each leg; otherwise the day is skipped.
    use_log_returns : bool
        Use log returns if True; otherwise use simple percentage returns.

    Returns
    -------
    pd.Series
        Daily PnL series (executed on the next trading day after the signal date).
    """
    # --- Precompute returns ---
    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.sort_index()
    rets = np.log(px).diff() if use_log_returns else px.pct_change()
    rets = rets.dropna(how="all")

    # --- Helper: strictly next trading day (> d) ---
    r_index = rets.index
    def _next_trading_day(d: pd.Timestamp):
        pos = r_index.searchsorted(pd.Timestamp(d), side="right")
        if pos >= len(r_index):
            return None
        return r_index[pos]

    # --- Main loop over signal dates ---
    daily_returns, backtest_dates = [], []
    L = lead_lag_matrix_rolling.copy()
    L.index = pd.to_datetime(L.index)
    L = L.sort_index()
    L = L[~L.index.duplicated(keep="last")]  # keep the last if duplicates exist

    for d in L.index:
        M = L.loc[d]

        # Skip if the matrix is all-NaN
        if isinstance(M, pd.Series):
            all_nan = M.isna().all()
        else:
            all_nan = M.isnull().all().all()
        if all_nan:
            continue

        # Asset score = row-wise mean (how much an asset tends to lead/lag others)
        lead_scores = M.mean(axis=1) if isinstance(M, pd.DataFrame) else M

        # Split scores into positive and negative sides
        pos_scores = lead_scores[lead_scores > 0]
        neg_scores = lead_scores[lead_scores < 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            # Cannot define leaders/followers per current rule; skip the day
            continue

        # Tail thresholds
        top_thresh = pos_scores.quantile(1 - quantile)   # leaders from positive tail
        bottom_thresh = neg_scores.quantile(quantile)    # followers from negative tail

        leaders = lead_scores[lead_scores >= top_thresh].index
        followers = lead_scores[lead_scores <= bottom_thresh].index

        # Minimum count filter
        if len(leaders) < min_names or len(followers) < min_names:
            continue

        # Optional inversion
        if invert:
            leaders, followers = followers, leaders

        # Execute on the next trading day
        next_day = _next_trading_day(d)
        if next_day is None:
            continue

        r_next = rets.loc[next_day]
        long_ret  = r_next[followers].dropna().mean() if len(followers) > 0 else 0.0
        short_ret = r_next[leaders].dropna().mean()   if len(leaders) > 0 else 0.0

        daily_returns.append(long_ret - short_ret)
        backtest_dates.append(next_day)

    return pd.Series(daily_returns, index=backtest_dates,
                     name=f"leadlag_hedge_{'inv' if invert else 'norm'}")


