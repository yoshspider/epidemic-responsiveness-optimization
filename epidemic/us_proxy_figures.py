"""County choropleths and comparisons for conditional U.S. proxy experiments."""
from pathlib import Path
import io
import zipfile
import numpy as np
import pandas as pd
import shapefile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm


def project(points, region):
    lon = np.asarray(points)[:, 0]
    lat = np.asarray(points)[:, 1]
    if region == 'AK':
        lon = np.where(lon > 0, lon - 360, lon)
        return np.c_[(lon + 154) * np.cos(np.deg2rad(64)), lat - 64]
    if region == 'HI':
        return np.c_[(lon + 157) * np.cos(np.deg2rad(20)), lat - 20]
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon + 96)
    phi1, phi2 = np.deg2rad([29.5, 45.5])
    factor = (np.sin(phi1) + np.sin(phi2)) / 2
    constant = np.cos(phi1)**2 + 2 * factor * np.sin(phi1)
    rho = 6371 * np.sqrt(constant - 2 * factor * np.sin(phi)) / factor
    phi0 = np.deg2rad(37.5)
    rho0 = 6371 * np.sqrt(constant - 2 * factor * np.sin(phi0)) / factor
    return np.c_[rho * np.sin(factor * lam), rho0 - rho * np.cos(factor * lam)]


def geometry():
    result = {'US': [], 'AK': [], 'HI': []}
    with zipfile.ZipFile('data/raw/census_counties_2020.zip') as archive:
        shp = next(n for n in archive.namelist() if n.endswith('.shp'))
        dbf = next(n for n in archive.namelist() if n.endswith('.dbf'))
        reader = shapefile.Reader(
            shp=io.BytesIO(
                archive.read(shp)), dbf=io.BytesIO(
                archive.read(dbf)))
        fields = [f[0] for f in reader.fields[1:]]
        for item in reader.iterShapeRecords():
            record = dict(zip(fields, item.record))
            state = record['STATEFP']
            if state in ['60', '66', '69', '72', '78']:
                continue
            region = 'AK' if state == '02' else 'HI' if state == '15' else 'US'
            fips = record['GEOID']
            fips = '02261' if fips in ['02063', '02066'] else fips
            boundaries = list(item.shape.parts) + [len(item.shape.points)]
            paths = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                points = project(item.shape.points[start:end], region)
                codes = np.full(len(points), MplPath.LINETO)
                codes[0] = MplPath.MOVETO
                codes[-1] = MplPath.CLOSEPOLY
                paths.append(MplPath(points, codes))
            result[region].append((fips, MplPath.make_compound_path(*paths)))
    return result


def draw(ax, geo, values, norm, cmap='YlOrRd', insets=True):
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('#dddddd')

    def region(target, key):
        patches = [PathPatch(path) for fips, path in geo[key]]
        color = np.ma.masked_invalid([values.get(fips, np.nan) for fips, path in geo[key]])
        collection = PatchCollection(
            patches,
            cmap=cmap,
            norm=norm,
            linewidth=.06,
            edgecolor='#ffffff')
        collection.set_array(color)
        collection.set_clim(norm.vmin, norm.vmax)
        target.add_collection(collection)
        target.set_aspect('equal')
        target.set_axis_off()
        return collection
    collection = region(ax, 'US')
    ax.set_xlim(-2500, 2450)
    ax.set_ylim(-1500, 1700)
    if insets:
        alaska = ax.inset_axes([0, .005, .18, .24])
        region(alaska, 'AK')
        alaska.set_xlim(-18, 13)
        alaska.set_ylim(-14, 9)
        hawaii = ax.inset_axes([.18, .005, .11, .16])
        region(hawaii, 'HI')
        hawaii.set_xlim(-5, 3)
        hawaii.set_ylim(-2, 3)
    return collection


def mapping(counties, values): return dict(zip(counties.fips, values))


def main():
    root = Path('outputs/us_proxy_study')
    out = root / 'figures'
    out.mkdir(exist_ok=True)
    geo = geometry()
    folder = root / 'geographic'
    counties = pd.read_csv(folder / 'counties.csv', dtype={'fips': str})
    n = counties.population.to_numpy()
    proxies = pd.read_csv(
        folder / 'proxy_values.csv',
        dtype={
            'fips': str}).set_index('fips').reindex(
        counties.fips)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    for ax, metric, title in zip(axes, ['mask', 'vaccine'], [
                                 'NYT/Dynata masking, July2020', 'HHS vaccine non-hesitancy, May–June2021']):
        collection = draw(ax, geo, mapping(counties, proxies[metric]), Normalize(.3, 1), 'viridis')
        ax.set_title(title, fontsize=12)
    fig.colorbar(collection, ax=axes, fraction=.02, pad=.01, label='Survey-based county fraction')
    fig.suptitle('Nationwide proxy inputs: 3,142 counties', fontsize=16)
    fig.text(.08, .04, 'NYT values are smoothed survey estimates; HHS values are modeled small-area estimates. Neither directly measures spending responsiveness.', fontsize=9)
    fig.savefig(out / 'national_proxy_inputs.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    base = np.load(folder / 'mask_distribution.npz')
    pred = np.load(folder / 'mask_demographic_prediction.npz')
    oracle = np.load(folder / 'mask_proxy.npz')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    norm = SymLogNorm(linthresh=10, vmin=0, vmax=100000)
    for row, result in enumerate([base, pred]):
        for col, week in enumerate([3, 7, 11]):
            collection = draw(axes[row, col], geo, mapping(
                counties, result['cumulative'][week] / n * 100000), norm)
            if row == 0:
                axes[row, col].set_title(f'Week {week + 1}', fontsize=12)
    fig.subplots_adjust(left=.06, right=.90, top=.84, bottom=.08, wspace=.01, hspace=.01)
    fig.suptitle('Spread with and without predicted masking information', fontsize=16, y=.98)
    fig.text(.5, .925, 'Controlled epidemic on U.S. geography; assumed responsiveness equals July2020 mask score; same intervention budget', ha='center', fontsize=10)
    fig.text(.015, .66, 'Distribution\nonly', rotation=90, va='center', fontsize=12)
    fig.text(.015, .29, 'Predicted\nmasking', rotation=90, va='center', fontsize=12)
    cax = fig.add_axes([.92, .20, .015, .54])
    bar = fig.colorbar(collection, cax=cax, ticks=[0, 10, 100, 1000, 10000, 100000])
    bar.set_label('Cumulative modeled infections per 100,000', fontsize=10)
    fig.text(.07, .025, 'Past-demographic forest predictions use held-out states. Synthetic initial epidemic states; actual Census populations and weekly GeoDS networks.', fontsize=9)
    fig.savefig(out / 'mask_spread_weeks_4_8_12.png', dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1))
    arrays = [base['cumulative'][-1] - oracle['cumulative']
              [-1], base['cumulative'][-1] - pred['cumulative'][-1]]
    rates = [a / n * 100000 for a in arrays]
    limit = max(1, float(np.quantile(np.abs(np.concatenate(rates)), .99)))
    norm = SymLogNorm(linthresh=.1, vmin=-limit, vmax=limit)
    for ax, values, raw, title in zip(
            axes, rates, arrays, ['Measured county mask map', 'Held-out demographic predictor']):
        collection = draw(ax, geo, mapping(counties, values), norm, 'RdBu')
        ax.set_title(f'{title}\n{raw.sum():,.0f} fewer modeled infections', fontsize=12)
    fig.colorbar(collection, ax=axes, fraction=.02, pad=.01,
                 label='Additional infections avoided per100,000 (blue=fewer; red=more)')
    fig.suptitle('Where county information changes allocation outcomes', fontsize=16)
    fig.text(.07, .04, 'Controlled U.S.-geography epidemic, 12weeks, central budget. Color scale capped at99th percentile; totals include every county.', fontsize=9)
    fig.savefig(out / 'mask_county_information_benefit.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    # Historical replay is always displayed separately from geographic stress test.
    real = root / 'real_fall'
    counties = pd.read_csv(real / 'counties.csv', dtype={'fips': str})
    n = counties.population.to_numpy()
    base = np.load(real / 'distribution.npz')
    pred = np.load(real / 'demographic_prediction.npz')
    difference = (base['cumulative'][-1] - pred['cumulative'][-1]) / n * 100000
    limit = max(1, float(np.quantile(np.abs(difference), .99)))
    fig, ax = plt.subplots(figsize=(10, 6))
    collection = draw(
        ax, geo, mapping(
            counties, difference), SymLogNorm(
            linthresh=.1, vmin=-limit, vmax=limit), 'RdBu')
    fig.colorbar(collection, ax=ax, fraction=.025, pad=.01,
                 label='Additional modeled infections avoided per100,000')
    ax.set_title(
        'Historical replay: predicted masking versus distribution-only allocation',
        fontsize=13)
    fig.text(.07, .055, 'September7–November29,2020; contemporaneously fitted transmission; hypothetical sustained intervention.\nNYT July masking assumed to represent responsiveness. Gray counties excluded from case initialization. Not a causal estimate.', fontsize=9)
    fig.savefig(out / 'historical_mask_prediction_benefit.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    data = pd.read_csv(folder / 'summary.csv')
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, metric, policy, title in zip(axes.ravel(), ['mask', 'mask', 'vaccine', 'vaccine'], ['proxy', 'demographic_prediction', 'proxy', 'past_mask_rank'], [
                                         'Mask map', 'Predicted masking', 'Vaccine map', 'Past mask ranking → vaccine attitudes']):
        table = data[(data.metric == metric) & (data.policy == policy) & (data.mobility == .15)].pivot(
            index='beta', columns='average_dose', values='improvement_percent').sort_index()
        limit = max(1, np.abs(table.to_numpy()).max())
        image = ax.imshow(
            table.to_numpy(),
            cmap='RdBu',
            norm=TwoSlopeNorm(
                0,
                vmin=-limit,
                vmax=limit),
            aspect='auto')
        for row in range(len(table)):
            for col in range(len(table.columns)):
                ax.text(col, row, f'{table.iloc[row, col]:.1f}%', ha='center', va='center', fontsize=11, color='white' if abs(
                    table.iloc[row, col]) > .65 * limit else 'black')
        ax.set_xticks(range(len(table.columns)), [str(x) for x in table.columns])
        ax.set_yticks(range(len(table)), [str(x) for x in table.index])
        ax.set_xlabel('Average intervention dose')
        ax.set_ylabel('Transmission β')
        ax.set_title(title, fontsize=12)
    fig.suptitle('Information benefit across the full parameter grid', fontsize=15)
    fig.text(.08, .02, 'Entries: % fewer infections than distribution-only allocation; negative values mean worse. Mobility share15%.\nThese are sensitivity scenarios, not calibrated real intervention effects or statistical confidence intervals.', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .96])
    fig.savefig(out / 'parameter_sensitivity.png', dpi=170)
    plt.close(fig)
    print('Saved national proxy maps, spread snapshots, benefit maps and sensitivity figure', flush=True)


if __name__ == '__main__':
    main()
