# Data sources and attribution

Raw sources are downloaded separately and retain their source-specific terms. No individual tweet records or large raw archives are included in Git.

The active replay uses NYT county cases and masking, Census populations, GeoDS mobility, and CDC SVI. The remaining sources below document earlier experiments.

| Source | Role | Documentation |
|---|---|---|
| The New York Times COVID county data | Historical case initialization and retrospective transmission calibration | https://github.com/nytimes/covid-19-data |
| The New York Times and Dynata | July 2–14, 2020 mask-use estimates for 3,142 counties | https://github.com/nytimes/covid-19-data/tree/master/mask-use |
| HHS/ASPE and CDC | Survey-based modeled county vaccine-hesitancy estimates, May 26–June 7, 2021 | https://catalog.data.gov/dataset/vaccine-hesitancy-for-covid-19-county-and-local-estimates |
| U.S. Census | Vintage 2019 county populations and 2020 cartographic boundaries | https://www.census.gov/programs-surveys/popest/data/data-sets.html |
| GeoDS | Actual weekly county-to-county population flows | https://github.com/GeoDS/COVID19USFlows-WeeklyFlows |
| CMU Delphi CTIS | Monthly descriptive mask/willingness estimates and behavioral prediction targets | https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html |
| CDC SVI 2018 | Pre-pandemic ACS socioeconomic covariates | https://www.atsdr.cdc.gov/place-health/php/svi/ |
| TBCOV / CrisisNLP | Sentiment and resident-location labels for county aggregation | https://crisisnlp.qcri.org/tbcov |

## Proxy definitions

- NYT mask score: `ALWAYS + FREQUENTLY + 0.5*SOMETIMES`.
- HHS vaccine non-hesitancy: `1 - Estimated hesitant`. This is not identical to direct vaccine acceptance or causal campaign responsiveness.
- CTIS monthly values are descriptive averages of overlapping public daily estimates, with explicit minimum-day coverage. They are not independent pooled county-month survey estimates.

NYT/Dynata uses geographic smoothing of responses. HHS uses Household Pulse Survey relationships, ACS microdata and a PUMA-to-county crosswalk. Nationwide estimated coverage must not be described as direct survey measurement in every county.

TBCOV files processed for the multi-month study cover August 2020–March 2021. Generic sentiment features are not a completed mask-topic classifier. No hydration is performed; ambiguous county labels are excluded.

## Chronology and geography

The historical September–November 2020 replay uses the preceding July mask survey. HHS 2021 attitudes are used in separately labeled geographic stress tests, not backdated into a prospective 2020 policy. Common historical mobility networks in those stress tests supply a controlled transport structure, not a calendar-matched 2021 forecast.

Historical case initialization distinguishes never-separately-reported geographies from counties with no cases before their first report. Population and boundary vintages differ for some Alaska county equivalents; maps explicitly associate the two 2020 subdivisions with the corresponding 2019 combined county when needed.

Download URLs, timestamps, hashes, coverage, exclusions and interpretation appear in the acquisition and experiment manifests. Data attribution should include **The New York Times and Dynata** for mask-use outputs, alongside the other sources used.
