import pandas as pd
import numpy as np


def _missing_stats(d: pd.DataFrame, cols):
    per_feat = d[cols].isna().mean().sort_values(ascending=False)
    overall = per_feat.mean()
    return overall, per_feat


def _fit_fill_stats(train_ts_df: pd.DataFrame, cols, global_stat: str):
    if train_ts_df is None:
        raise ValueError("Please provide train_ts_df or fill_stats directly (training set statistics) to avoid data leakage.")
    if global_stat == 'mean':
        return train_ts_df[cols].mean()
    elif global_stat == 'median':
        return train_ts_df[cols].median()
    else:
        raise ValueError("global_stat must be either 'mean' or 'median'.")


def impute_ts_ehr_global(
    ts_df: pd.DataFrame,
    features_columns,
    train_ts_df: pd.DataFrame = None,
    id_col: str = 'hadm_id',
    time_col: str = 'timepoint',
    global_stat: str = 'mean',          # 'mean' or 'median'
    fill_stats: pd.Series = None,       # pass training set statistics directly (used in val/test)
    verbose: bool = True,
    copy: bool = True
):
    """
    Fill all missing values at once using only the global statistics from the training set.
    Returns: (filled_df, overall_dict, per_feature_table, used_fill_stats)
    """
    df = ts_df.copy() if copy else ts_df

    # Step 0: initial missing ratio
    overall0, perfeat0 = _missing_stats(df, features_columns)

    # compute / obtain fill values
    if fill_stats is None:
        fill_stats = _fit_fill_stats(train_ts_df, features_columns, global_stat)

    # Step 1: fill with global statistics
    df[features_columns] = df[features_columns].fillna(fill_stats)
    overall1, perfeat1 = _missing_stats(df, features_columns)

    if verbose:
        print(f"[Init]        overall missing ratio: {overall0:.4f}")
        print(f"[Global-{global_stat}] overall missing ratio: {overall1:.4f}")

    per_feature_table = pd.DataFrame({
        'init': perfeat0,
        f'after_global_{global_stat}': perfeat1
    })
    overall_dict = {
        'init': overall0,
        f'after_global_{global_stat}': overall1
    }
    return df, overall_dict


def impute_ts_ehr_linear_interp(
    ts_df: pd.DataFrame,
    features_columns,
    train_ts_df: pd.DataFrame = None,
    id_col: str = 'hadm_id',
    time_col: str = 'timepoint',
    global_stat: str = 'mean',           # 'mean' or 'median'
    fill_stats: pd.Series = None,        # pass training set statistics directly (used in val/test)
    interp_method: str = 'linear',       # linear interpolation; use 'time' if datetime index
    interp_limit: int = None,            # max consecutive missing steps allowed for interpolation (prevents large gaps)
    interp_limit_direction: str = 'both',# 'forward'|'backward'|'both'
    interp_order: int = None,            # needed for polynomial / spline interpolation (e.g. 'polynomial'/'spline')
    verbose: bool = True,
    copy: bool = True
):
    """
    First sort by (id_col, time_col) and interpolate within each group, then fill remaining missing with training set global statistics.
    Note: 'linear' method is based on row order; to interpolate by real time, ensure time column is datetime and set method='time'.
    Returns: (filled_df, overall_dict, per_feature_table, used_fill_stats)
    """
    df = ts_df.copy() if copy else ts_df

    # Step 0: initial missing ratio
    overall0, perfeat0 = _missing_stats(df, features_columns)

    # sort to ensure correct time direction (critical for interpolation)
    if time_col in df.columns:
        df = df.sort_values([id_col, time_col])
    else:
        df = df.sort_values([id_col])

    # Step 1: group-wise interpolation
    interp_kwargs = dict(method=interp_method,
                         limit=interp_limit,
                         limit_direction=interp_limit_direction)
    # pass order only when needed to avoid parameter errors in some methods
    if interp_order is not None:
        interp_kwargs['order'] = interp_order

    df[features_columns] = (
        df.groupby(id_col, sort=False)[features_columns]
          .apply(lambda g: g.interpolate(**interp_kwargs))
          .reset_index(level=0, drop=True)
    )

    overall1, perfeat1 = _missing_stats(df, features_columns)

    # Step 2: fill remaining missing with global statistics
    if fill_stats is None:
        fill_stats = _fit_fill_stats(train_ts_df, features_columns, global_stat)

    df[features_columns] = df[features_columns].fillna(fill_stats)
    overall2, perfeat2 = _missing_stats(df, features_columns)

    if verbose:
        print(f"[Init]        overall missing ratio: {overall0:.4f}")
        print(f"[Interpolate] overall missing ratio: {overall1:.4f}")
        print(f"[Global-{global_stat}] overall missing ratio: {overall2:.4f}")

    overall_dict = {
        'init': overall0,
        'after_interpolate': overall1,
        f'after_global_{global_stat}': overall2
    }
    return df, overall_dict


def impute_ts_ehr_bffill(
    ts_df: pd.DataFrame,
    features_columns,
    train_ts_df: pd.DataFrame = None,   # used to fit global statistics; can be None if fill_stats provided
    id_col: str = 'hadm_id',
    time_col: str = 'timepoint',
    global_stat: str = 'mean',          # 'mean' or 'median'
    fill_stats: pd.Series = None,       # pass pre-computed global statistics (avoids leakage in val/test)
    verbose: bool = True,
    copy: bool = True
):
    """
    Perform in sequence: groupby(id) forward fill -> backward fill -> fill with training set global statistics.
    Returns: (filled df, dict of overall missing ratios at each stage, DataFrame of per-feature missing ratios at each stage, global statistics used)

    Note: ensure that features_columns in ts_df are numeric (or can be filled by mean/median).
    """
    df = ts_df.copy() if copy else ts_df

    # —— Step 0: initial missing ratio
    overall0, perfeat0 = _missing_stats(df, features_columns)

    # —— sort to ensure correct time direction (critical)
    if time_col in df.columns:
        df = df.sort_values([id_col, time_col])
    else:
        df = df.sort_values([id_col])

    # —— Step 1: forward fill within group
    df[features_columns] = df.groupby(id_col, sort=False)[features_columns].ffill()
    overall1, perfeat1 = _missing_stats(df, features_columns)

    # —— Step 2: backward fill within group
    df[features_columns] = df.groupby(id_col, sort=False)[features_columns].bfill()
    overall2, perfeat2 = _missing_stats(df, features_columns)

    # —— Step 3: fill remaining missing with training set statistics
    if fill_stats is None:
        if train_ts_df is None:
            raise ValueError("Please provide train_ts_df or fill_stats directly (training set statistics) to avoid data leakage.")
        if global_stat == 'mean':
            fill_stats = train_ts_df[features_columns].mean()
        elif global_stat == 'median':
            fill_stats = train_ts_df[features_columns].median()
        else:
            raise ValueError("global_stat must be either 'mean' or 'median'.")

    df[features_columns] = df[features_columns].fillna(fill_stats)
    overall3, perfeat3 = _missing_stats(df, features_columns)

    if verbose:
        print(f"[Init]      overall missing ratio: {overall0:.4f}")
        print(f"[FFill]     overall missing ratio: {overall1:.4f}")
        print(f"[BFill]     overall missing ratio: {overall2:.4f}")
        print(f"[Global-{global_stat}] overall missing ratio: {overall3:.4f}")

        # uncomment next line to inspect timepoint distribution
        # if time_col in df.columns: print(df[time_col].value_counts())

    overall_dict = {
        'init': overall0,
        'after_ffill': overall1,
        'after_bfill': overall2,
        f'after_global_{global_stat}': overall3
    }

    return df, overall_dict
