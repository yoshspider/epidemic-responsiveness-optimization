# Candidate empirical data sources

Sources checked October 13, 2026; no source is silently merged into the simulation.

- [NYT county mask survey and documentation](https://github.com/nytimes/covid-19-data/tree/master/mask-use). Direct CSV has county survey response shares; choose and document a proxy formula such as FREQUENTLY+ALWAYS. It measures reported adherence, not causal marginal response to spending.
- [NYT historical COVID county reports](https://github.com/nytimes/covid-19-data). Public historical case/death tables can inform calibration; examine reporting artifacts and geography exceptions.
- [Daily county sentiment, dataset author's description](https://www.patrickbaylis.com/posts/2025-02-22-tweet-data/). Author describes county/day sentiment aggregates spanning 2014–2022. These are general Twitter sentiment, not COVID-specific sentiment. They could support a separate baseline after inspecting access, license and county coverage; they cannot substitute for COVID-specific features without changing the research question.
- [Twitter Sentiment Geographical Index dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10562363/). Describes aggregated geographic daily indices and links a public data archive. Verify temporal coverage, geography and topic selection before use. Raw tweet methods requiring hydration do not satisfy the project's preferred input restrictions.

## Selected sources after user clarification

The intended sources are NYT cases, GeoDS weekly county mobility, Census population estimates, CTIS behavior targets, NYT mask survey for independent comparison, and TBCOV sentiment/location labels. General sentiment sources above are background alternatives only.

Downloaded files and SHA256 provenance are in `data/raw/downloads.json`: NYT 2020 cases and mask survey; Census Vintage 2019 county totals; GeoDS week beginning July 6, 2020; CTIS September 8 and October 2020 public daily mask summaries. Census Vintage 2019 is a documented period-appropriate choice, not a claim to identify the unspecified 387 KB CSV exactly. GeoDS's downloaded file has 624,225 rows.

NYT has 884,737 rows, including 8,266 without FIPS and 13,436 negative revisions among rows with FIPS. Flags remain in transformed tables. Census and NYT mask FIPS sets match at 3,142 counties. `data/transformed/audit.json` records checks. Census totals do not provide age65 counts or intervention prices; both remain needed.

CTIS weighted mask signal begins September 8, 2020 according to [official signal documentation](https://cmu-delphi.github.io/delphi-epidata/api/covidcast-signals/fb-survey.html). It cannot train a prospective July 6 policy. October public summaries cover only 557 county identifiers after excluding state aggregates ending 000, with 1–31 observed days. Missing counties are not imputed. The transformed file is a descriptive average of overlapping daily seven-day estimates, **not** a direct monthly pooled survey estimator. Review [CTIS archive monthly aggregates](https://doi.org/10.3886/ICPSR39207.v1) and source geography to resolve target coverage.

NYT survey is July 2–14, 2020. CTIS comparison later in 2020 measures cross-period geographic agreement, not simultaneous validation. CTIS issue dates may be later than observation dates; prospective work must use values published by the decision cutoff.

[TBCOV base-release descriptors](https://crisisnlp.qcri.org/tbcov) include sentiment and location labels usable without hydration. [Official repository](https://github.com/CrisisComputing/TBCOV) distinguishes country/language ID lists from base files with labels. The full September archive was downloaded and streamed: 88,931,846 records, 8,937,269 mapped US county records. State/county names are matched through an explicit Census crosswalk; ambiguous matches are excluded. Real held-out predictions and downstream comparisons are in `outputs/tbcov_prediction` and `outputs/tbcov_policy`; see `EVALUATION_REPORT.md`. The source permits non-commercial research.

The brief's NYT behavior summary matches `ALWAYS + FREQUENTLY + 0.5*SOMETIMES` (mean 0.776477, SD 0.111308). The new corrected-proxy replay is in `outputs/graded_proxy_replay`. Earlier `ALWAYS + FREQUENTLY` runs are retained as separate sensitivity analyses. Both are descriptive adherence proxies, not identified causal response parameters.

Weekly network support is implemented; only one actual week is downloaded. Empirical runs require complete weekly coverage. Lookahead freezes the current network. GeoDS week-start filenames summarize that entire week: use lagged completed weeks for prospective decisions or label same-week flows retrospective.

No empirical policy results are available. Remaining inputs: complete weekly networks, age counts, explicit costs, usable CTIS target coverage, TBCOV base files/crosswalk and a temporally valid experiment window.
