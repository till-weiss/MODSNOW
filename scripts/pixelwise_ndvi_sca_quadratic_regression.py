#!/usr/bin/env python3
"""
Prototype: pixel-wise annual NDVI regression with linear and quadratic SCA terms.

This script helps test potential non-linear relationships by fitting, per pixel:
  Linear:    y = b0 + b1*x
  Quadratic: y = a0 + a1*x + a2*x^2

Usage example:
  python scripts/pixelwise_ndvi_sca_quadratic_regression.py \
    --ndvi_nc_path /Users/tillweiss/Desktop/MODSNOW/data/NDVI_nc/NDVI_clean.nc \
    --x0 1000 --y0 1000 --nx 50 --ny 50 \
    --ndvi_doy_start 200 --ndvi_doy_end 260 \
    --sca_doy_start 140 --sca_doy_end 180 \
    --out_nc_path pixelwise_ndvi_sca_quadratic_annual.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_MASTER_CSV = "/Users/tillweiss/Desktop/MODSNOW/data/snow_aggregated/ndvi_sca_master.csv"


def _detect_dims(da: xr.DataArray) -> tuple[str, str, str]:
    if "time" not in da.dims:
        raise ValueError(f"NDVI variable '{da.name}' must contain a 'time' dimension; found dims={da.dims}.")
    spatial_dims = [d for d in da.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected exactly 2 spatial dimensions plus time, got dims={da.dims}.")
    y_dim, x_dim = spatial_dims
    return "time", y_dim, x_dim


def _build_annual_ndvi(ndvi: xr.DataArray, doy_start: int, doy_end: int) -> xr.DataArray:
    doy = ndvi["time"].dt.dayofyear
    ndvi_window = ndvi.where((doy >= doy_start) & (doy <= doy_end))
    return ndvi_window.groupby("time.year").mean("time", skipna=True).sortby("year")


def _load_annual_sca(master_csv_path: str, doy_start: int, doy_end: int) -> pd.Series:
    df = pd.read_csv(master_csv_path)
    required = {"Year", "DOY", "Snow_16d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in SCA CSV: {sorted(missing)}")
    df = df[(df["DOY"] >= doy_start) & (df["DOY"] <= doy_end)].copy()
    if df.empty:
        raise ValueError(f"No SCA rows remain after DOY filter [{doy_start}, {doy_end}] from {master_csv_path}.")
    return df.groupby("Year", as_index=True)["Snow_16d"].mean().sort_index()


def _clip_start(start: int, size: int, max_size: int) -> int:
    return int(np.clip(start, 0, max(0, max_size - size)))


def _linear_ols(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_b = x[:, None, None]
    valid = np.isfinite(y) & np.isfinite(x_b)
    n = valid.sum(axis=0)

    x_valid = np.where(valid, x_b, np.nan)
    y_valid = np.where(valid, y, np.nan)

    mean_x = np.nanmean(x_valid, axis=0)
    mean_y = np.nanmean(y_valid, axis=0)

    xm = x_valid - mean_x[None, :, :]
    ym = y_valid - mean_y[None, :, :]

    cov_xy = np.nansum(xm * ym, axis=0)
    var_x = np.nansum(xm * xm, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x

    y_hat = intercept[None, :, :] + slope[None, :, :] * x_b
    resid = np.where(valid, y - y_hat, np.nan)
    sse = np.nansum(resid * resid, axis=0)
    sst = np.nansum((y_valid - mean_y[None, :, :]) ** 2, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - (sse / sst)

    invalid = (n < min_n) | (var_x == 0) | (sst == 0) | (~np.isfinite(var_x)) | (~np.isfinite(sst))
    slope = np.where(invalid, np.nan, slope)
    intercept = np.where(invalid, np.nan, intercept)
    r2 = np.where(invalid, np.nan, r2)
    return slope, intercept, r2, n.astype(np.int16)


def _quadratic_ols(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a0, a1, a2, r2, n for y = a0 + a1*x + a2*x^2 per pixel."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    t, ny, nx = y.shape
    npix = ny * nx
    y2d = y.reshape(t, npix)

    valid = np.isfinite(y2d) & np.isfinite(x[:, None])
    n = valid.sum(axis=0)

    a0 = np.full(npix, np.nan, dtype=float)
    a1 = np.full(npix, np.nan, dtype=float)
    a2 = np.full(npix, np.nan, dtype=float)
    r2 = np.full(npix, np.nan, dtype=float)

    eligible = np.where(n >= min_n)[0]
    if eligible.size == 0:
        return a0.reshape(ny, nx), a1.reshape(ny, nx), a2.reshape(ny, nx), r2.reshape(ny, nx), n.reshape(ny, nx).astype(np.int16)

    for idx in eligible:
        m = valid[:, idx]
        xi = x[m]
        yi = y2d[m, idx]

        if xi.size < 3:
            continue

        X = np.column_stack([np.ones_like(xi), xi, xi * xi])
        beta, _, _, _ = np.linalg.lstsq(X, yi, rcond=None)

        yhat = X @ beta
        sse = np.sum((yi - yhat) ** 2)
        ybar = np.mean(yi)
        sst = np.sum((yi - ybar) ** 2)
        if sst <= 0:
            continue

        a0[idx], a1[idx], a2[idx] = beta
        r2[idx] = 1.0 - (sse / sst)

    return (
        a0.reshape(ny, nx),
        a1.reshape(ny, nx),
        a2.reshape(ny, nx),
        r2.reshape(ny, nx),
        n.reshape(ny, nx).astype(np.int16),
    )


def run(args: argparse.Namespace) -> None:
    ds = xr.open_dataset(args.ndvi_nc_path)
    if args.ndvi_var not in ds.data_vars:
        available = ", ".join(map(str, ds.data_vars))
        raise KeyError(f"Variable '{args.ndvi_var}' not found in {args.ndvi_nc_path}. Available variables: {available}")

    ndvi = ds[args.ndvi_var]
    _, y_dim, x_dim = _detect_dims(ndvi)

    ny_all = ndvi.sizes[y_dim]
    nx_all = ndvi.sizes[x_dim]
    x0 = _clip_start(args.x0, args.nx, nx_all)
    y0 = _clip_start(args.y0, args.ny, ny_all)

    ndvi_tile = ndvi.isel({y_dim: slice(y0, y0 + args.ny), x_dim: slice(x0, x0 + args.nx)})
    annual_ndvi = _build_annual_ndvi(ndvi_tile, args.ndvi_doy_start, args.ndvi_doy_end)

    sca_annual = _load_annual_sca(args.master_csv_path, args.sca_doy_start, args.sca_doy_end)

    ndvi_years = annual_ndvi["year"].values.astype(int)
    sca_years = sca_annual.index.values.astype(int)
    years = np.intersect1d(ndvi_years, sca_years)
    years = np.sort(years)

    if years.size < args.min_n:
        raise ValueError(f"Only {years.size} intersected years found; need at least min_n={args.min_n}.")

    annual_ndvi = annual_ndvi.sel(year=years)
    x = sca_annual.loc[years].to_numpy(dtype=float)
    y = np.asarray(annual_ndvi.to_numpy(), dtype=float)

    lin_slope, lin_intercept, lin_r2, lin_n = _linear_ols(x=x, y=y, min_n=args.min_n)
    q_a0, q_a1, q_a2, q_r2, q_n = _quadratic_ols(x=x, y=y, min_n=args.min_n)

    delta_r2 = q_r2 - lin_r2

    out_ds = xr.Dataset(
        data_vars={
            "linear_slope": ((y_dim, x_dim), lin_slope),
            "linear_intercept": ((y_dim, x_dim), lin_intercept),
            "linear_r2": ((y_dim, x_dim), lin_r2),
            "linear_n": ((y_dim, x_dim), lin_n),
            "quad_intercept": ((y_dim, x_dim), q_a0),
            "quad_x": ((y_dim, x_dim), q_a1),
            "quad_x2": ((y_dim, x_dim), q_a2),
            "quad_r2": ((y_dim, x_dim), q_r2),
            "quad_n": ((y_dim, x_dim), q_n),
            "delta_r2_quad_minus_linear": ((y_dim, x_dim), delta_r2),
        },
        coords={y_dim: annual_ndvi[y_dim], x_dim: annual_ndvi[x_dim]},
        attrs={
            "ndvi_var": args.ndvi_var,
            "ndvi_doy_start": args.ndvi_doy_start,
            "ndvi_doy_end": args.ndvi_doy_end,
            "sca_doy_start": args.sca_doy_start,
            "sca_doy_end": args.sca_doy_end,
            "x0": int(x0),
            "y0": int(y0),
            "nx": int(args.nx),
            "ny": int(args.ny),
            "min_n": int(args.min_n),
            "years_used": ",".join(map(str, years.tolist())),
        },
    )

    out_path = Path(args.out_nc_path)
    out_ds.to_netcdf(out_path)

    valid_lin = np.isfinite(lin_r2)
    valid_quad = np.isfinite(q_r2)
    valid_both = valid_lin & valid_quad
    improved = valid_both & (delta_r2 > 0)

    print("Run summary")
    print(f"  tile (x0,y0,nx,ny): ({x0}, {y0}, {args.nx}, {args.ny})")
    print(f"  intersected years: {years.size}")
    print(f"  pixels valid linear r2: {np.mean(valid_lin):.3f}")
    print(f"  pixels valid quadratic r2: {np.mean(valid_quad):.3f}")
    print(f"  pixels where quadratic improves r2: {np.mean(improved):.3f}")
    if np.any(valid_both):
        print(f"  delta r2 (quad-linear) median over valid: {np.nanmedian(delta_r2[valid_both]):.4f}")
    print(f"  wrote: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pixel-wise linear and quadratic NDVI~SCA regression prototype")
    parser.add_argument("--ndvi_nc_path", required=True)
    parser.add_argument("--ndvi_var", default="ndvi_clean")
    parser.add_argument("--master_csv_path", default=DEFAULT_MASTER_CSV)

    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)

    parser.add_argument("--ndvi_doy_start", type=int, default=200)
    parser.add_argument("--ndvi_doy_end", type=int, default=260)
    parser.add_argument("--sca_doy_start", type=int, default=140)
    parser.add_argument("--sca_doy_end", type=int, default=180)
    parser.add_argument("--min_n", type=int, default=8)

    parser.add_argument("--out_nc_path", default="pixelwise_ndvi_sca_quadratic_annual.nc")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
