#!/usr/bin/env python3
"""
Prototype: pixel-wise annual NDVI~SCA regressions for altitude-band SCA predictors.

Usage example:
python scripts/altitude_band_ndvi_sca_regression.py \
  --ndvi_nc_path /Users/tillweiss/Desktop/MODSNOW/data/NDVI_nc/NDVI_clean.nc \
  --master_csv_path /Users/tillweiss/Desktop/MODSNOW/data/snow_aggregated/ndvi_sca_master.csv \
  --ds 2 \
  --x0 500 --y0 500 --nx 50 --ny 50 \
  --ndvi_doy_start 200 --ndvi_doy_end 260 \
  --sca_doy_start 140 --sca_doy_end 180 \
  --out_nc_path altitude_band_pixelwise_ndvi_sca_ds2.nc
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_MASTER_CSV = "/Users/tillweiss/Desktop/MODSNOW/data/snow_aggregated/ndvi_sca_master.csv"
BAND_REGEX = re.compile(r"^col_\d+_16d$")


def _detect_dims(da: xr.DataArray) -> tuple[str, str, str]:
    if "time" not in da.dims:
        raise ValueError(f"NDVI variable '{da.name}' must contain a 'time' dimension; found dims={da.dims}.")

    spatial_dims = [d for d in da.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected NDVI dims to be (time, y, x) or (time, lat, lon); found dims={da.dims}.")

    y_dim, x_dim = spatial_dims
    return "time", y_dim, x_dim


def _downsample_ndvi(ndvi: xr.DataArray, y_dim: str, x_dim: str, ds_factor: int) -> xr.DataArray:
    if ds_factor < 1:
        raise ValueError(f"--ds must be >=1, got {ds_factor}")
    if ds_factor == 1:
        return ndvi
    return ndvi.coarsen({y_dim: ds_factor, x_dim: ds_factor}, boundary="trim").mean()


def _clip_start(start: int, size: int, max_size: int) -> int:
    max_start = max(0, max_size - size)
    return int(np.clip(start, 0, max_start))


def _build_annual_ndvi(ndvi: xr.DataArray, ndvi_doy_start: int, ndvi_doy_end: int) -> xr.DataArray:
    doy = ndvi["time"].dt.dayofyear
    ndvi_window = ndvi.where((doy >= ndvi_doy_start) & (doy <= ndvi_doy_end))
    return ndvi_window.groupby("time.year").mean("time", skipna=True).sortby("year")


def _parse_band_columns(df: pd.DataFrame, band_cols_arg: str | None) -> list[str]:
    if band_cols_arg:
        requested = [c.strip() for c in band_cols_arg.split(",") if c.strip()]
        if not requested:
            raise ValueError("--band_cols was provided but no valid column names were parsed.")
        missing = [c for c in requested if c not in df.columns]
        if missing:
            raise ValueError(f"Requested --band_cols not found in CSV: {missing}")
        return requested

    auto = [c for c in df.columns if BAND_REGEX.match(c)]
    if not auto:
        raise ValueError("No altitude-band columns matching regex '^col_\\d+_16d$' were found in CSV.")
    return sorted(auto)


def _load_annual_band_predictors(
    master_csv_path: str,
    sca_doy_start: int,
    sca_doy_end: int,
    band_cols_arg: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(master_csv_path)
    required_base = {"Year", "DOY"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise ValueError(f"Missing required columns in master CSV: {sorted(missing_base)}")

    band_cols = _parse_band_columns(df, band_cols_arg)

    df = df[(df["DOY"] >= sca_doy_start) & (df["DOY"] <= sca_doy_end)].copy()
    if df.empty:
        raise ValueError(
            f"No rows remain in CSV after DOY filter [{sca_doy_start}, {sca_doy_end}] from {master_csv_path}."
        )

    grouped = df.groupby("Year", as_index=True)[band_cols].mean().sort_index()
    return grouped, band_cols


def _pixelwise_ols(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # x shape: (t,), y shape: (t, ny, nx)
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

    invalid = (n < min_n) | (~np.isfinite(var_x)) | (var_x == 0) | (~np.isfinite(sst)) | (sst == 0)
    slope = np.where(invalid, np.nan, slope)
    intercept = np.where(invalid, np.nan, intercept)
    r2 = np.where(invalid, np.nan, r2)

    return slope, intercept, r2, n.astype(np.int16)


def run(args: argparse.Namespace) -> None:
    ds = xr.open_dataset(args.ndvi_nc_path)
    if args.ndvi_var not in ds.data_vars:
        available = ", ".join(map(str, ds.data_vars))
        raise KeyError(
            f"Variable '{args.ndvi_var}' not found in {args.ndvi_nc_path}. Available variables: {available}"
        )

    ndvi = ds[args.ndvi_var]
    _, y_dim, x_dim = _detect_dims(ndvi)

    ndvi = _downsample_ndvi(ndvi, y_dim=y_dim, x_dim=x_dim, ds_factor=args.ds)

    _, y_dim_ds, x_dim_ds = _detect_dims(ndvi)
    ny_all = ndvi.sizes[y_dim_ds]
    nx_all = ndvi.sizes[x_dim_ds]
    x0 = _clip_start(args.x0, args.nx, nx_all)
    y0 = _clip_start(args.y0, args.ny, ny_all)

    ndvi_tile = ndvi.isel({y_dim_ds: slice(y0, y0 + args.ny), x_dim_ds: slice(x0, x0 + args.nx)})
    annual_ndvi = _build_annual_ndvi(ndvi_tile, args.ndvi_doy_start, args.ndvi_doy_end)

    band_year_df, band_cols = _load_annual_band_predictors(
        master_csv_path=args.master_csv_path,
        sca_doy_start=args.sca_doy_start,
        sca_doy_end=args.sca_doy_end,
        band_cols_arg=args.band_cols,
    )

    ndvi_years = annual_ndvi["year"].values.astype(int)
    sca_years = band_year_df.index.values.astype(int)
    years = np.intersect1d(ndvi_years, sca_years)
    years = np.sort(years)

    if years.size < args.min_n:
        raise ValueError(
            f"Only {years.size} intersected years between NDVI and band predictors after filtering; need at least min_n={args.min_n}."
        )

    annual_ndvi = annual_ndvi.sel(year=years)
    band_year_df = band_year_df.loc[years, band_cols]

    y = np.asarray(annual_ndvi.to_numpy(), dtype=float)
    nbands = len(band_cols)
    ny = y.shape[1]
    nx = y.shape[2]

    slope_all = np.full((nbands, ny, nx), np.nan, dtype=float)
    intercept_all = np.full((nbands, ny, nx), np.nan, dtype=float)
    r2_all = np.full((nbands, ny, nx), np.nan, dtype=float)
    n_all = np.zeros((nbands, ny, nx), dtype=np.int16)

    usable_mask = np.isfinite(y).sum(axis=0) >= args.min_n

    for b_idx, band_name in enumerate(band_cols):
        x = band_year_df[band_name].to_numpy(dtype=float)

        # Apply per-band finite predictor filtering as extra safety.
        finite_x = np.isfinite(x)
        if np.count_nonzero(finite_x) < args.min_n:
            n_counts = np.isfinite(y[finite_x]).sum(axis=0)
            n_all[b_idx] = n_counts.astype(np.int16)
            continue

        slope, intercept, r2, n = _pixelwise_ols(x=x[finite_x], y=y[finite_x], min_n=args.min_n)
        slope_all[b_idx] = slope
        intercept_all[b_idx] = intercept
        r2_all[b_idx] = r2
        n_all[b_idx] = n

    coords = {
        "band": band_cols,
        y_dim_ds: annual_ndvi[y_dim_ds],
        x_dim_ds: annual_ndvi[x_dim_ds],
    }

    out_ds = xr.Dataset(
        data_vars={
            "slope": (("band", y_dim_ds, x_dim_ds), slope_all),
            "intercept": (("band", y_dim_ds, x_dim_ds), intercept_all),
            "r2": (("band", y_dim_ds, x_dim_ds), r2_all),
            "n": (("band", y_dim_ds, x_dim_ds), n_all),
        },
        coords=coords,
        attrs={
            "ndvi_var": args.ndvi_var,
            "ndvi_doy_start": args.ndvi_doy_start,
            "ndvi_doy_end": args.ndvi_doy_end,
            "sca_doy_start": args.sca_doy_start,
            "sca_doy_end": args.sca_doy_end,
            "ds": args.ds,
            "x0": int(x0),
            "y0": int(y0),
            "nx": int(args.nx),
            "ny": int(args.ny),
            "min_n": int(args.min_n),
            "band_cols_used": ",".join(band_cols),
            "years_used": ",".join(map(str, years.tolist())),
        },
    )

    out_path = Path(args.out_nc_path)
    out_ds.to_netcdf(out_path)

    print("Run summary")
    print(f"  bands detected: {len(band_cols)}")
    print(f"  first bands: {band_cols[:5]}")
    print(f"  intersected years: {years.size}")
    print(f"  output shapes -> slope: {slope_all.shape}, r2: {r2_all.shape}, n: {n_all.shape}")
    print(f"  usable-pixel fraction (n_valid_ndvi_years >= {args.min_n}): {float(np.mean(usable_mask)):.3f}")

    for b_idx, band_name in enumerate(band_cols):
        valid_px = n_all[b_idx] >= args.min_n
        frac_valid = float(np.mean(valid_px))
        r2_valid = r2_all[b_idx][np.isfinite(r2_all[b_idx]) & valid_px]
        med_r2 = float(np.nanmedian(r2_valid)) if r2_valid.size > 0 else np.nan
        print(f"  {band_name}: frac_pixels_n>=min_n={frac_valid:.3f}, median_r2={med_r2:.3f}")

    print(f"  wrote: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Altitude-band pixel-wise annual NDVI~SCA regression prototype")
    parser.add_argument("--ndvi_nc_path", required=True)
    parser.add_argument("--ndvi_var", default="ndvi_clean")
    parser.add_argument("--master_csv_path", default=DEFAULT_MASTER_CSV)

    parser.add_argument("--x0", type=int, default=0)
    parser.add_argument("--y0", type=int, default=0)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)
    parser.add_argument("--ds", type=int, default=1)

    parser.add_argument("--ndvi_doy_start", type=int, default=200)
    parser.add_argument("--ndvi_doy_end", type=int, default=260)
    parser.add_argument("--sca_doy_start", type=int, default=140)
    parser.add_argument("--sca_doy_end", type=int, default=180)

    parser.add_argument("--min_n", type=int, default=8)
    parser.add_argument("--band_cols", default=None, help="Comma-separated list, e.g. col_1_16d,col_2_16d")
    parser.add_argument("--out_nc_path", default="altitude_band_ndvi_sca_regression_annual.nc")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
