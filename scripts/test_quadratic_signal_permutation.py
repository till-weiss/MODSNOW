#!/usr/bin/env python3
"""Test whether pixel-wise quadratic NDVI~SCA gains are likely real using permutation nulls.

This script:
1) Summarizes quadratic-regression outputs from an existing NetCDF.
2) Recomputes sampled pixel annual NDVI time series from raw NDVI netCDF.
3) Recomputes annual SCA predictor from master CSV.
4) Compares observed ΔR² (quadratic-linear) vs a permutation null that shuffles SCA across years.
5) Writes diagnostic PNGs into --out_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _detect_spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    spatial_dims = [d for d in da.dims if d != "time" and d != "year"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected 2 spatial dims; got dims={da.dims}")
    return spatial_dims[0], spatial_dims[1]


def _autodetect_ndvi_var(ds: xr.Dataset, ndvi_var: str | None) -> str:
    if ndvi_var:
        if ndvi_var not in ds.data_vars:
            raise KeyError(f"Requested --ndvi_var '{ndvi_var}' not found. Available: {list(ds.data_vars)}")
        return ndvi_var

    for cand in ["ndvi_clean", "ndvi", "NDVI", "NDVI_clean"]:
        if cand in ds.data_vars:
            return cand

    for name, da in ds.data_vars.items():
        if "time" in da.dims and da.ndim == 3 and "ndvi" in name.lower():
            return name

    raise ValueError(
        "Could not autodetect NDVI variable. Please pass --ndvi_var. "
        f"Available variables: {list(ds.data_vars)}"
    )


def _annual_ndvi_from_raw(
    ndvi_nc_path: str,
    ndvi_var: str | None,
    x0: int,
    y0: int,
    nx: int,
    ny: int,
    ndvi_doy_start: int,
    ndvi_doy_end: int,
) -> tuple[xr.DataArray, str, str]:
    ds = xr.open_dataset(ndvi_nc_path)
    var = _autodetect_ndvi_var(ds, ndvi_var)
    ndvi = ds[var]

    if "time" not in ndvi.dims:
        raise ValueError(f"NDVI variable '{var}' must have a time dimension; got {ndvi.dims}")

    y_dim, x_dim = _detect_spatial_dims(ndvi)

    ny_all = ndvi.sizes[y_dim]
    nx_all = ndvi.sizes[x_dim]
    x0 = int(np.clip(x0, 0, max(0, nx_all - nx)))
    y0 = int(np.clip(y0, 0, max(0, ny_all - ny)))

    ndvi_tile = ndvi.isel({y_dim: slice(y0, y0 + ny), x_dim: slice(x0, x0 + nx)})

    doy = ndvi_tile["time"].dt.dayofyear
    ndvi_win = ndvi_tile.where((doy >= ndvi_doy_start) & (doy <= ndvi_doy_end), drop=True)
    if ndvi_win.sizes.get("time", 0) == 0:
        raise ValueError(
            f"No NDVI timesteps remain in DOY window [{ndvi_doy_start}, {ndvi_doy_end}] for selected tile."
        )

    ndvi_ann = ndvi_win.groupby("time.year").mean("time", skipna=True).sortby("year")
    return ndvi_ann, y_dim, x_dim


def _load_sca_annual(
    master_csv_path: str,
    sca_doy_start: int,
    sca_doy_end: int,
    sca_year_col: str | None,
    sca_doy_col: str | None,
    sca_value_col: str | None,
) -> pd.Series:
    df = pd.read_csv(master_csv_path)

    year_col = sca_year_col
    doy_col = sca_doy_col
    value_col = sca_value_col

    if year_col is None:
        year_col = "Year" if "Year" in df.columns else ("year" if "year" in df.columns else None)
    if doy_col is None:
        doy_col = "DOY" if "DOY" in df.columns else ("doy" if "doy" in df.columns else None)
    if value_col is None:
        value_col = "Snow_16d" if "Snow_16d" in df.columns else None

    if year_col is None or doy_col is None or value_col is None:
        raise ValueError(
            "Could not identify SCA columns. Expected defaults like ['Year'/'year', 'DOY'/'doy', 'Snow_16d'] "
            "or pass explicit flags --sca_year_col --sca_doy_col --sca_value_col. "
            f"CSV columns are: {list(df.columns)}"
        )

    missing = [c for c in [year_col, doy_col, value_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Provided SCA column(s) not found: {missing}. CSV columns are: {list(df.columns)}"
        )

    df = df[(df[doy_col] >= sca_doy_start) & (df[doy_col] <= sca_doy_end)].copy()
    if df.empty:
        raise ValueError(
            f"No rows remain after SCA DOY filter [{sca_doy_start}, {sca_doy_end}] in {master_csv_path}."
        )

    s = df.groupby(year_col, as_index=True)[value_col].mean().sort_index()
    s.index = s.index.astype(int)
    return s


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y.size < 2:
        return np.nan
    ybar = np.mean(y)
    sst = np.sum((y - ybar) ** 2)
    if sst <= 0:
        return np.nan
    sse = np.sum((y - yhat) ** 2)
    return 1.0 - (sse / sst)


def _fit_delta_r2_single(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < min_n:
        return np.nan, np.nan

    xg = x[mask]
    yg = y[mask]
    if xg.size < 3:
        return np.nan, np.nan

    X_lin = np.column_stack([xg, np.ones_like(xg)])
    b_lin, _, _, _ = np.linalg.lstsq(X_lin, yg, rcond=None)
    yhat_lin = X_lin @ b_lin
    r2_lin = _r2(yg, yhat_lin)

    X_quad = np.column_stack([xg * xg, xg, np.ones_like(xg)])
    b_quad, _, _, _ = np.linalg.lstsq(X_quad, yg, rcond=None)
    yhat_quad = X_quad @ b_quad
    r2_quad = _r2(yg, yhat_quad)

    if not np.isfinite(r2_lin) or not np.isfinite(r2_quad):
        return np.nan, np.nan
    return r2_lin, r2_quad - r2_lin


def _save_map(data2d: np.ndarray, title: str, out_path: Path, cmap: str, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(8, 5))
    plt.imshow(data2d, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ndvi{args.ndvi_doy_start}_{args.ndvi_doy_end}__sca{args.sca_doy_start}_{args.sca_doy_end}"

    qds = xr.open_dataset(args.quad_nc_path)
    required_vars = [
        "linear_r2",
        "linear_n",
        "quad_r2",
        "quad_n",
        "quad_x2",
        "delta_r2_quad_minus_linear",
    ]
    missing = [v for v in required_vars if v not in qds.data_vars]
    if missing:
        raise ValueError(f"Quadratic NetCDF missing required vars: {missing}. Available: {list(qds.data_vars)}")

    delta = qds["delta_r2_quad_minus_linear"].to_numpy()
    lin_r2 = qds["linear_r2"].to_numpy()
    quad_r2 = qds["quad_r2"].to_numpy()
    quad_x2 = qds["quad_x2"].to_numpy()
    quad_n = qds["quad_n"].to_numpy()

    valid_r2 = np.isfinite(lin_r2) & np.isfinite(quad_r2)
    valid_delta = np.isfinite(delta)

    frac_valid_r2 = float(np.mean(valid_r2))
    frac_delta_gt0 = float(np.mean(valid_delta & (delta > 0.00)))
    frac_delta_gt002 = float(np.mean(valid_delta & (delta > 0.02)))
    frac_delta_gt005 = float(np.mean(valid_delta & (delta > 0.05)))

    delta_valid_vals = delta[valid_delta]
    delta_mean = float(np.nanmean(delta_valid_vals)) if delta_valid_vals.size > 0 else np.nan
    delta_median = float(np.nanmedian(delta_valid_vals)) if delta_valid_vals.size > 0 else np.nan

    strong_mask = valid_delta & (delta > 0.05)
    frac_hump = float(np.mean(quad_x2[strong_mask] < 0)) if np.any(strong_mask) else np.nan

    ydim, xdim = _detect_spatial_dims(qds["delta_r2_quad_minus_linear"])
    q_ny = qds.sizes[ydim]
    q_nx = qds.sizes[xdim]

    ndvi_ann, ndvi_ydim, ndvi_xdim = _annual_ndvi_from_raw(
        ndvi_nc_path=args.ndvi_nc_path,
        ndvi_var=args.ndvi_var,
        x0=args.x0,
        y0=args.y0,
        nx=args.nx,
        ny=args.ny,
        ndvi_doy_start=args.ndvi_doy_start,
        ndvi_doy_end=args.ndvi_doy_end,
    )

    if ndvi_ann.sizes[ndvi_ydim] != q_ny or ndvi_ann.sizes[ndvi_xdim] != q_nx:
        raise ValueError(
            "Tile shape mismatch between quadratic NetCDF and raw NDVI-derived annual array. "
            f"quad shape=({q_ny},{q_nx}), ndvi shape=({ndvi_ann.sizes[ndvi_ydim]},{ndvi_ann.sizes[ndvi_xdim]}). "
            "Ensure --x0/--y0/--nx/--ny and source files match the quadratic run."
        )

    sca_ann = _load_sca_annual(
        master_csv_path=args.master_csv_path,
        sca_doy_start=args.sca_doy_start,
        sca_doy_end=args.sca_doy_end,
        sca_year_col=args.sca_year_col,
        sca_doy_col=args.sca_doy_col,
        sca_value_col=args.sca_value_col,
    )

    years_ndvi = ndvi_ann["year"].values.astype(int)
    years_sca = sca_ann.index.values.astype(int)
    years = np.intersect1d(years_ndvi, years_sca)
    years = np.sort(years)
    if years.size < args.min_n:
        raise ValueError(
            f"Only {years.size} intersected years between annual NDVI and SCA; need at least min_n={args.min_n}."
        )

    ndvi_ann = ndvi_ann.sel(year=years)
    x_true = sca_ann.loc[years].to_numpy(dtype=float)
    y_all = np.asarray(ndvi_ann.to_numpy(), dtype=float)  # (t, y, x)

    sample_mask = np.isfinite(delta) & (quad_n >= args.min_n)
    sample_iy, sample_ix = np.where(sample_mask)
    n_valid = sample_iy.size
    if n_valid == 0:
        raise ValueError("No valid pixels available for permutation sampling (mask: finite delta and quad_n >= min_n).")

    rng = np.random.default_rng(args.seed)
    n_sample = int(min(args.n_pixels, n_valid))
    pick = rng.choice(n_valid, size=n_sample, replace=False)
    sy = sample_iy[pick]
    sx = sample_ix[pick]

    y_sample = y_all[:, sy, sx]  # (t, n_sample)
    linear_r2_sample = lin_r2[sy, sx]

    obs_delta = np.full(n_sample, np.nan, dtype=float)
    obs_lin = np.full(n_sample, np.nan, dtype=float)
    for i in range(n_sample):
        r2_lin_i, delta_i = _fit_delta_r2_single(x_true, y_sample[:, i], min_n=args.min_n)
        obs_lin[i] = r2_lin_i
        obs_delta[i] = delta_i

    perm_delta_all = []
    null_medians = np.full(args.n_perm, np.nan, dtype=float)
    for p in range(args.n_perm):
        x_perm = rng.permutation(x_true)
        perm_delta = np.full(n_sample, np.nan, dtype=float)
        for i in range(n_sample):
            _, d = _fit_delta_r2_single(x_perm, y_sample[:, i], min_n=args.min_n)
            perm_delta[i] = d
        perm_delta_all.append(perm_delta)
        null_medians[p] = np.nanmedian(perm_delta)

    perm_delta_all = np.concatenate(perm_delta_all)

    obs_median = float(np.nanmedian(obs_delta))
    obs_mean = float(np.nanmean(obs_delta))
    null_median = float(np.nanmedian(perm_delta_all))
    null_mean = float(np.nanmean(perm_delta_all))

    p_emp = float((1 + np.sum(null_medians >= obs_median)) / (args.n_perm + 1))

    _save_map(
        data2d=delta,
        title="ΔR² (quadratic - linear)",
        out_path=out_dir / f"map_delta_r2_{stem}.png",
        cmap="YlGn",
        vmin=0,
        vmax=0.3,
    )

    curv_map = np.where(delta > 0.05, quad_x2, np.nan)
    if np.any(np.isfinite(curv_map)):
        p1, p99 = np.nanpercentile(curv_map, [1, 99])
        lim = max(abs(p1), abs(p99))
    else:
        lim = 1.0
    _save_map(
        data2d=curv_map,
        title="quad_x2 where ΔR² > 0.05",
        out_path=out_dir / f"map_quad_x2_masked_{stem}.png",
        cmap="RdBu",
        vmin=-lim,
        vmax=lim,
    )

    plt.figure(figsize=(7, 4))
    plt.hist(delta_valid_vals, bins=100, color="steelblue", alpha=0.85)
    plt.title("Histogram of ΔR² over valid pixels")
    plt.xlabel("ΔR² (quadratic - linear)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_delta_r2_all_{stem}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.hist(delta_valid_vals, bins=100, color="steelblue", alpha=0.85)
    plt.xlim(0, 0.3)
    plt.title("Histogram of ΔR² over valid pixels (zoom)")
    plt.xlabel("ΔR² (quadratic - linear)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_delta_r2_all_zoom_{stem}.png", dpi=200)
    plt.close()

    curv_vals = quad_x2[strong_mask]
    plt.figure(figsize=(7, 4))
    plt.hist(curv_vals[np.isfinite(curv_vals)], bins=100, color="darkorange", alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1)
    plt.title("quad_x2 over pixels with ΔR² > 0.05")
    plt.xlabel("quad_x2")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / f"hist_quad_x2_strong_delta_{stem}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.hist(obs_delta[np.isfinite(obs_delta)], bins=60, alpha=0.6, label="Observed ΔR² (sample)")
    plt.hist(perm_delta_all[np.isfinite(perm_delta_all)], bins=60, alpha=0.6, label="Permuted ΔR² (pooled)")
    plt.axvline(obs_median, color="blue", linestyle="-", linewidth=2, label=f"Obs median={obs_median:.4f}")
    plt.axvline(null_median, color="red", linestyle="--", linewidth=2, label=f"Null median={null_median:.4f}")
    plt.title("Observed vs permuted ΔR²")
    plt.xlabel("ΔR² (quadratic - linear)")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"perm_observed_vs_null_delta_r2_{stem}.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    sc = np.isfinite(obs_delta) & np.isfinite(linear_r2_sample)
    plt.scatter(linear_r2_sample[sc], obs_delta[sc], s=8, alpha=0.4)
    plt.xlabel("linear_r2 (sampled pixels)")
    plt.ylabel("observed ΔR²")
    plt.title("Where quadratic helps")
    plt.tight_layout()
    plt.savefig(out_dir / f"scatter_linear_r2_vs_observed_delta_{stem}.png", dpi=200)
    plt.close()

    print("=== Quadratic Signal Permutation Test ===")
    print(f"quad_nc_path: {args.quad_nc_path}")
    print(f"ndvi_nc_path: {args.ndvi_nc_path}")
    print(f"master_csv_path: {args.master_csv_path}")
    print(
        f"DOY windows -> NDVI [{args.ndvi_doy_start}, {args.ndvi_doy_end}], "
        f"SCA [{args.sca_doy_start}, {args.sca_doy_end}]"
    )
    print(f"tile -> x0={args.x0}, y0={args.y0}, nx={args.nx}, ny={args.ny}")
    print(f"intersected years: {years.size} ({years.min()}..{years.max()})")
    print(f"valid pixels in quadratic map mask: {n_valid}")
    print(f"sampled pixels: {n_sample}")

    print("\n-- Stats from quadratic NetCDF (Section A) --")
    print(f"fraction finite linear_r2 & quad_r2: {frac_valid_r2:.4f}")
    print(f"fraction delta_r2 > 0.00: {frac_delta_gt0:.4f}")
    print(f"fraction delta_r2 > 0.02: {frac_delta_gt002:.4f}")
    print(f"fraction delta_r2 > 0.05: {frac_delta_gt005:.4f}")
    print(f"delta_r2 median (valid): {delta_median:.6f}")
    print(f"delta_r2 mean (valid): {delta_mean:.6f}")
    print(f"among delta_r2>0.05, fraction quad_x2<0: {frac_hump:.4f}")

    print("\n-- Permutation stats (Section B) --")
    print(f"observed delta_r2 median (sample): {obs_median:.6f}")
    print(f"observed delta_r2 mean (sample): {obs_mean:.6f}")
    print(f"null delta_r2 median (pooled perms): {null_median:.6f}")
    print(f"null delta_r2 mean (pooled perms): {null_mean:.6f}")
    print(f"empirical p-value (null medians >= observed median): {p_emp:.6f}")
    print(f"plots written to: {out_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Permutation test for quadratic NDVI~SCA signal")
    p.add_argument("--quad_nc_path", required=True)
    p.add_argument("--ndvi_nc_path", required=True)
    p.add_argument("--master_csv_path", required=True)

    p.add_argument("--ndvi_var", default=None)

    p.add_argument("--x0", type=int, default=0)
    p.add_argument("--y0", type=int, default=0)
    p.add_argument("--nx", type=int, default=1601)
    p.add_argument("--ny", type=int, default=1081)

    p.add_argument("--ndvi_doy_start", type=int, default=200)
    p.add_argument("--ndvi_doy_end", type=int, default=260)
    p.add_argument("--sca_doy_start", type=int, default=140)
    p.add_argument("--sca_doy_end", type=int, default=180)

    p.add_argument("--sca_year_col", default=None)
    p.add_argument("--sca_doy_col", default=None)
    p.add_argument("--sca_value_col", default=None)

    p.add_argument("--min_n", type=int, default=8)
    p.add_argument("--n_pixels", type=int, default=5000)
    p.add_argument("--n_perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    return p


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
