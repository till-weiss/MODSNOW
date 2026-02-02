import re
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import stats
import pingouin as pg
import numpy as np
import seaborn as sns
import pandas as pd
import requests as r
import getpass, pprint, time, os, json
import geopandas as gpd
from pathlib import Path


def process_snow(
    snow_path: str,
    ndvi: pd.DataFrame,
    out_csv: str,
    base_cols=("Year", "Day", "Snow_Mean"),
    window_days: int = 16
) -> pd.DataFrame:
    # --- sanity ---
    if not isinstance(ndvi, pd.DataFrame):
        raise TypeError("Pass the NDVI DataFrame (ndvi), not the path.")
    if "Date" not in ndvi.columns:
        raise ValueError("ndvi must contain a 'Date' column.")

    ndvi_dates = pd.Index(
        pd.to_datetime(ndvi["Date"])
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    # --- read snow ---
    snow = pd.read_csv(snow_path, delim_whitespace=True)

    # rename snow columns: base + numbered extras (Snow_Percent becomes col_1, etc.)
    ncols = snow.shape[1]
    if ncols < len(base_cols):
        raise ValueError(
            f"Snow file has {ncols} columns but base_cols expects {len(base_cols)}."
        )

    snow.columns = list(base_cols) + [
        f"col_{i}" for i in range(1, ncols - len(base_cols) + 1)
    ]

    # date column
    snow["Date"] = (
        pd.to_datetime(snow["Year"].astype(int).astype(str), format="%Y")
        + pd.to_timedelta(snow["Day"] - 1, unit="D")
    )

    # daily snow
    snow_daily = snow.set_index("Date").resample("D").asfreq()

    # ensure we can compute the forward window up to last NDVI date + window
    full_index = pd.date_range(
        start=ndvi_dates.min(),
        end=ndvi_dates.max() + pd.Timedelta(days=window_days - 1),
        freq="D",
    )
    snow_d = snow_daily.reindex(full_index)

    # --- compute 16d means for ALL col_* columns ---
    col_cols = [c for c in snow_d.columns if c.startswith("col_")]

    # output only at NDVI dates (NO NDVI COLUMN)
    out = snow_d.loc[ndvi_dates].copy()

    # add col_*_16d for each numbered column (includes former Snow_Percent as col_1_16d)
    for c in col_cols:
        out[f"{c}_16d"] = [
            snow_d.loc[d: d + pd.Timedelta(days=window_days - 1), c].mean()
            for d in ndvi_dates
        ]

    # time fields
    out = out.reset_index().rename(columns={"index": "Date"})
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["DOY"] = out["Date"].dt.dayofyear

    # save all columns in 'out'
    out.to_csv(out_csv, index=False)
    return out


def get_modis_ndvi(shp, dir, name):

    os.chdir(dir)    

    user = 'weiss14'     
    password = "bemcog-bixqe3-bykpaZ"
    api = 'https://appeears.earthdatacloud.nasa.gov/api/'  

    token_response = r.post('{}login'.format(api), auth=(user, password)).json() 
    token = token_response['token']                      
    head = {'Authorization': 'Bearer {}'.format(token)}

    nps = gpd.read_file(shp)

    nps_gc = nps[nps['id']==0]  
    nps_gc = nps_gc[['geometry']].to_json() 
    nps_gc = json.loads(nps_gc) 

    projections = r.get('{}spatial/proj'.format(api)).json()  
    projs = {}                                 
    for p in projections: projs[p['Name']] = p 


    #Set task configurations
    task_name = f'NDVI_{task_name}'
    task_type = 'area'
    proj = projs['geographic']['Name']
    outFormat = ['geotiff', 'netcdf4']
    recurring = False

    # Use a single continuous date range (API-safe)
    startDate = "03-01-2000"
    endDate   = "08-31-2025"

    # ----------------------------
    # Task definition
    # ----------------------------
    task = {
        'task_type': task_type,
        'task_name': task_name,
        'params': {
            'dates': [{
                'startDate': startDate,
                'endDate': endDate
            }],
            'layers': [
                {
                    'product': 'MOD13Q1.061',
                    'layer': '_250m_16_days_NDVI'
                }
            ],
            'qualityFilters': {
                'MOD13Q1.061': {
                    'SummaryQA': [0]
                }
            },
            'output': {
                'format': {'type': outFormat[1]},  # netcdf4
                'projection': proj
            },
            'geo': nps_gc
        }
    }


    # ----------------------------
    # Submit task
    # ----------------------------
    task_response = r.post(f'{api}task', json=task, headers=head).json()

    if 'task_id' not in task_response:
        raise RuntimeError(f"Task submission failed: {task_response}")

    task_id = task_response['task_id']
    print(task_id)


    # ----------------------------
    # Monitor task
    # ----------------------------
    starttime = time.time()
    while r.get(f'{api}task/{task_id}', headers=head).json()['status'] != 'done':
        status = r.get(f'{api}task/{task_id}', headers=head).json()['status']
        print(status)
        time.sleep(20.0 - ((time.time() - starttime) % 20.0))


    # ----------------------------
    # Download results
    # ----------------------------
    destDir = os.path.join(dir, task_name)
    os.makedirs(destDir, exist_ok=True)
    bundle = r.get(f'{api}bundle/{task_id}', headers=head).json()

    files = {f['file_id']: f['file_name'] for f in bundle['files']}

    for file_id, fname in files.items():
        dl = r.get(
            f'{api}bundle/{task_id}/{file_id}',
            headers=head,
            stream=True,
            allow_redirects=True
        )

        filename = fname.split('/')[-1]
        filepath = os.path.join(destDir, filename)

        with open(filepath, 'wb') as out:
            for chunk in dl.iter_content(chunk_size=8192):
                out.write(chunk)

    print(f"Download complete. Files are in:\n{destDir}")




def plot_ndvi_timeseries(ndvi_snow, label, out_dir = str | Path | None):

    if out_dir is None:
        out_dir = Path('plots')         
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    df_corr = ndvi_snow[['NDVI', 'Snow_16d']].dropna()

    x = df_corr['Snow_16d']
    y = df_corr['NDVI']

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    y_pred = intercept + slope * x
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))

    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = intercept + slope * x_fit

    plt.figure(figsize=(8,6), dpi=300)
    plt.scatter(x, y, alpha=0.6, label='Data')
    plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Linear regression')

    plt.xlabel('Snow cover (16-day mean) [%]')
    plt.ylabel('NDVI')
    plt.title(f'NDVI vs. Snow cover {label}')
    plt.grid(True)

    n_obs = len(df_corr)

    plt.text(
        0.95, 0.95,
        f'NDVI = {slope:.4f} × SCA + {intercept:.4f}\n'
        f'n = {n_obs}\n'
        f'$r$ = {r_value:.3f}; '
        f'$r^2$ = {r_value**2:.3f}; '
        f'RMSE = {rmse:.4f}',
        transform=plt.gca().transAxes,
        va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = out_dir / f'ndvi_scatter_{safe_label}.png'


    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    pearson_coef, pearson_p = stats.pearsonr(x, y)
    spearman_coef, spearman_p = stats.spearmanr(x, y)

    print(f'Pearson correlation: {pearson_coef:.3f}, p-value = {pearson_p:.3e}')
    print(f'Spearman correlation: {spearman_coef:.3f}, p-value = {spearman_p:.3e}')

    print(f'Regression slope: {slope:.4f}')
    print(f'Regression intercept: {intercept:.4f}')
    print(f'Regression p-value: {p_value:.3e}')


def compute_cor_heatmap(ndvi_snow, label, out_dir = str | Path | None):

    if out_dir is None:
        out_dir = Path('plots')         
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ndvi_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='NDVI')
    snow_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='Snow_16d')

    corr_matrix = pd.DataFrame(
    index=ndvi_wide.columns,
    columns=snow_wide.columns,
    dtype=float
    )

    for ndvi_doy in ndvi_wide.columns:
        for snow_doy in snow_wide.columns:
            x = ndvi_wide[ndvi_doy]
            y = snow_wide[snow_doy]

            valid = x.notna() & y.notna()

            if valid.sum() >= 5:  
                corr_matrix.loc[ndvi_doy, snow_doy] = x[valid].corr(y[valid])
            else:
                corr_matrix.loc[ndvi_doy, snow_doy] = np.nan

    plt.figure(figsize=(16,12), dpi=300)
    sns.heatmap(corr_matrix, annot=True, cmap='Spectral').invert_yaxis()

    plt.xlabel('Snow cover DOY')
    plt.ylabel('NDVI DOY')
    plt.title(f'NDVI–Snow DOY × DOY Pearson Correlation (r) {label}')

    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = out_dir / f'ndvi_snow_cor_heatmap_{safe_label}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()    
    plt.close()


def stat_filter_heatmap(ndvi_snow, label, out_dir = str | Path | None):

    if out_dir is None:
        out_dir = Path('plots')         
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ndvi_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='NDVI')
    snow_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='Snow_16d')

    doys = ndvi_wide.columns

    r_mat = pd.DataFrame(index=doys, columns=doys, dtype=float)
    p_mat = pd.DataFrame(index=doys, columns=doys, dtype=float)
    ci_low_mat = pd.DataFrame(index=doys, columns=doys, dtype=float)
    ci_high_mat = pd.DataFrame(index=doys, columns=doys, dtype=float)

    for ndvi_doy in doys:
        for snow_doy in doys:
            res = pg.corr(
                ndvi_wide[ndvi_doy],
                snow_wide[snow_doy],
                method="pearson"
            )

            r_mat.loc[ndvi_doy, snow_doy] = res["r"].iloc[0]
            p_mat.loc[ndvi_doy, snow_doy] = res["p-val"].iloc[0]
            ci_low_mat.loc[ndvi_doy, snow_doy] = res["CI95%"].iloc[0][0]
            ci_high_mat.loc[ndvi_doy, snow_doy] = res["CI95%"].iloc[0][1]

            r_sig = r_mat.copy()


    r_sig[p_mat > 0.05] = np.nan

    plt.figure(figsize=(16,12), dpi=300)
    sns.heatmap(r_sig, annot=True, cmap='Spectral').invert_yaxis()

    plt.xlabel('Snow cover DOY')
    plt.ylabel('NDVI DOY')
    plt.title(f'Significant NDVI–Snow DOY × DOY Correlations (p ≤ 0.05) {label}')

    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = out_dir / f'heatmap_sig_{safe_label}.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    ci_significant = ~(
        (ci_low_mat <= 0) & (ci_high_mat >= 0)
    )

    r_ci_sig = r_mat.copy()
    r_ci_sig[~ci_significant] = np.nan

    plt.figure(figsize=(16,12))
    sns.heatmap(r_ci_sig, annot=True, cmap='Spectral').invert_yaxis()

    plt.xlabel('Snow cover DOY')
    plt.ylabel('NDVI DOY')
    plt.title(f'NDVI–Snow Correlations with 95% CI Not Crossing Zero {label}')

    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = out_dir / f'heatmap_ci_{safe_label}.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def compute_cor2_heatmap(ndvi_snow, label, out_dir = str | Path | None):

    if out_dir is None:
        out_dir = Path('plots')         
    else:
        out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    ndvi_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='NDVI')
    snow_wide = ndvi_snow.pivot(index='Year', columns='DOY', values='Snow_16d')

    corr2_matrix = pd.DataFrame(
        index=ndvi_wide.columns,
        columns=snow_wide.columns,
        dtype=float
    )

    for ndvi_doy in ndvi_wide.columns:
        for snow_doy in snow_wide.columns:
            x = ndvi_wide[ndvi_doy]
            y = snow_wide[snow_doy]

            valid = x.notna() & y.notna()

            if valid.sum() >= 5:  
                r = x[valid].corr(y[valid])
                corr2_matrix.loc[ndvi_doy, snow_doy] = r ** 2
                
            else:
                corr2_matrix.loc[ndvi_doy, snow_doy] = np.nan

    plt.figure(figsize=(16,12), dpi=300)
    sns.heatmap(corr2_matrix, annot=True, cmap='Spectral').invert_yaxis()

    plt.xlabel('Snow cover DOY')
    plt.ylabel('NDVI DOY')
    plt.title(f'NDVI–Snow DOY × DOY $r^2$ {label}')

    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', label)
    filename = out_dir / f'heatmap_corr2_{safe_label}.png'

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
