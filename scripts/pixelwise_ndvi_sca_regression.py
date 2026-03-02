#!/usr/bin/env python3
"""
Prototype: pixel-wise annual NDVI ~ regional SCA regressions.

Usage example:
  python scripts/pixelwise_ndvi_sca_regression.py \
    --ndvi_nc_path /path/to/NDVI_clean.nc \
    --x0 1000 --y0 1000 --nx 50 --ny 50 \
    --ndvi_doy_start 200 --ndvi_doy_end 260 \
    --sca_doy_start 140 --sca_doy_end 180 \
    --out_nc_path pixelwise_ndvi_sca_regression_annual.nc
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
        raise ValueError(
            f"Expected exactly 2 spatial dimensions plus time, got dims={da.dims}."
        )

    y_dim, x_dim = spatial_dims
    return "time", y_dim, x_dim


def _build_annual_ndvi(
    ndvi: xr.DataArray,
    ndvi_doy_start: int,
    ndvi_doy_end: int,
) -> xr.DataArray:
    doy = ndvi["time"].dt.dayofyear
    doy_mask = (doy >= ndvi_doy_start) & (doy <= ndvi_doy_end)
    ndvi_window = ndvi.where(doy_mask)
    annual = ndvi_window.groupby("time.year").mean("time", skipna=True)
    return annual


def _clip_tile_start(start: int, size: int, max_size: int) -> int:
    max_start = max(0, max_size - size)
    return int(np.clip(start, 0, max_start))


def _tile_usable_fraction(usable_mask: np.ndarray, x0: int, y0: int, nx: int, ny: int) -> float:
    tile = usable_mask[y0 : y0 + ny, x0 : x0 + nx]
    return float(np.mean(tile))


def _choose_tile(
    usable_mask: np.ndarray,
    x0: int,
    y0: int,
    nx: int,
    ny: int,
    min_tile_valid_frac: float,
    search_radius_px: int,
    step_px: int,
    do_search: bool,
) -> tuple[int, int, float]:
    h, w = usable_mask.shape
    x0 = _clip_tile_start(x0, nx, w)
    y0 = _clip_tile_start(y0, ny, h)

    initial_frac = _tile_usable_fraction(usable_mask, x0, y0, nx, ny)
    if (not do_search) or (initial_frac >= min_tile_valid_frac):
        return x0, y0, initial_frac

    best_x0, best_y0, best_frac = x0, y0, initial_frac

    x_candidates = range(x0 - search_radius_px, x0 + search_radius_px + 1, step_px)
    y_candidates = range(y0 - search_radius_px, y0 + search_radius_px + 1, step_px)

    for yc in y_candidates:
        ycs = _clip_tile_start(yc, ny, h)
        for xc in x_candidates:
            xcs = _clip_tile_start(xc, nx, w)
            frac = _tile_usable_fraction(usable_mask, xcs, ycs, nx, ny)

            if frac > best_frac:
                best_x0, best_y0, best_frac = xcs, ycs, frac
            elif np.isclose(frac, best_frac):
                old_dist = np.hypot(best_x0 - x0, best_y0 - y0)
                new_dist = np.hypot(xcs - x0, ycs - y0)
                if new_dist < old_dist:
                    best_x0, best_y0, best_frac = xcs, ycs, frac

    return best_x0, best_y0, best_frac


def _load_annual_sca(
    master_csv_path: str,
    sca_doy_start: int,
    sca_doy_end: int,
) -> pd.Series:
    df = pd.read_csv(master_csv_path)

    required = {"Year", "DOY", "Snow_16d"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in SCA CSV: {sorted(missing)}")

    df = df[(df["DOY"] >= sca_doy_start) & (df["DOY"] <= sca_doy_end)].copy()
    if df.empty:
        raise ValueError(
            f"No SCA rows remain after DOY filter [{sca_doy_start}, {sca_doy_end}] from {master_csv_path}."
        )

    annual_sca = df.groupby("Year", as_index=True)["Snow_16d"].mean().sort_index()
    return annual_sca


def _pixelwise_ols(x: np.ndarray, y: np.ndarray, min_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # x: (t,), y: (t, ny, nx)
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
        available = ", ".join(ds.data_vars)
        raise KeyError(
            f"Variable '{args.ndvi_var}' not found in {args.ndvi_nc_path}. Available variables: {available}"
        )

    ndvi = ds[args.ndvi_var]
    _, y_dim, x_dim = _detect_dims(ndvi)

    annual_ndvi = _build_annual_ndvi(ndvi, args.ndvi_doy_start, args.ndvi_doy_end)
    annual_ndvi = annual_ndvi.sortby("year")

    valid_counts = annual_ndvi.count(dim="year")
    usable_mask = (valid_counts >= args.min_n).values

    chosen_x0, chosen_y0, usable_fraction = _choose_tile(
        usable_mask=usable_mask,
        x0=args.x0,
        y0=args.y0,
        nx=args.nx,
        ny=args.ny,
        min_tile_valid_frac=args.min_tile_valid_frac,
        search_radius_px=args.search_radius_px,
        step_px=args.step_px,
        do_search=(not args.no_tile_search),
    )

    annual_tile = annual_ndvi.isel(
        {
            y_dim: slice(chosen_y0, chosen_y0 + args.ny),
            x_dim: slice(chosen_x0, chosen_x0 + args.nx),
        }
    )

    sca_annual = _load_annual_sca(args.master_csv_path, args.sca_doy_start, args.sca_doy_end)
    ndvi_years = annual_tile["year"].values.astype(int)
    sca_years = sca_annual.index.values.astype(int)

    years = np.intersect1d(ndvi_years, sca_years)
    years = np.sort(years)
    if years.size < args.min_n:
        raise ValueError(
            f"Only {years.size} intersected years between NDVI and SCA after filtering; need at least min_n={args.min_n}."
        )

    annual_tile = annual_tile.sel(year=years)
    x = sca_annual.loc[years].to_numpy(dtype=float)
    y = annual_tile.to_numpy(dtype=float)

    slope, intercept, r2, n = _pixelwise_ols(x=x, y=y, min_n=args.min_n)

    coords = {
        y_dim: annual_tile[y_dim],
        x_dim: annual_tile[x_dim],
    }

    out_ds = xr.Dataset(
        data_vars={
            "slope": ((y_dim, x_dim), slope),
            "intercept": ((y_dim, x_dim), intercept),
            "r2": ((y_dim, x_dim), r2),
            "n": ((y_dim, x_dim), n),
        },
        coords=coords,
        attrs={
            "ndvi_doy_start": args.ndvi_doy_start,
            "ndvi_doy_end": args.ndvi_doy_end,
            "sca_doy_start": args.sca_doy_start,
            "sca_doy_end": args.sca_doy_end,
            "min_n": args.min_n,
            "min_tile_valid_frac": args.min_tile_valid_frac,
            "chosen_x0": int(chosen_x0),
            "chosen_y0": int(chosen_y0),
            "usable_fraction": float(usable_fraction),
        },
    )

    out_path = Path(args.out_nc_path)
    out_ds.to_netcdf(out_path)

    valid_px = n >= args.min_n
    valid_r2 = r2[np.isfinite(r2) & valid_px]
    frac_valid_px = float(np.mean(valid_px))

    print("Run summary")
    print(f"  chosen tile x0,y0: ({chosen_x0}, {chosen_y0}), usable_fraction={usable_fraction:.3f}")
    print(f"  intersected years: {years.size}")
    print(f"  fraction pixels with n >= {args.min_n}: {frac_valid_px:.3f}")
    if valid_r2.size > 0:
        print(
            f"  r2 over valid pixels (min/median/max): "
            f"{np.nanmin(valid_r2):.3f} / {np.nanmedian(valid_r2):.3f} / {np.nanmax(valid_r2):.3f}"
        )
    else:
        print("  r2 over valid pixels: no valid pixels")
    print(f"  wrote: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pixel-wise annual NDVI~SCA regression prototype")
    parser.add_argument("--ndvi_nc_path", required=True)
    parser.add_argument("--ndvi_var", default="ndvi_clean")
    parser.add_argument("--master_csv_path", default=DEFAULT_MASTER_CSV)

    parser.add_argument("--x0", type=int, required=True)
    parser.add_argument("--y0", type=int, required=True)
    parser.add_argument("--nx", type=int, default=50)
    parser.add_argument("--ny", type=int, default=50)

    parser.add_argument("--ndvi_doy_start", type=int, default=200)
    parser.add_argument("--ndvi_doy_end", type=int, default=260)
    parser.add_argument("--sca_doy_start", type=int, default=140)
    parser.add_argument("--sca_doy_end", type=int, default=180)

    parser.add_argument("--min_n", type=int, default=8)
    parser.add_argument("--min_tile_valid_frac", type=float, default=0.70)
    parser.add_argument("--search_radius_px", type=int, default=200)
    parser.add_argument("--step_px", type=int, default=25)
    parser.add_argument("--no_tile_search", action="store_true")

    parser.add_argument("--out_nc_path", default="pixelwise_ndvi_sca_regression_annual.nc")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    run(parser.parse_args())
