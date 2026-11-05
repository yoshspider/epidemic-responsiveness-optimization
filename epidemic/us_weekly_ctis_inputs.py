"""Fetch the two CTIS inputs needed by the weekly proxy study and aggregate masking."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .acquire import download
from .prepare import ctis_monthly

ROOT = Path("data/raw")
CTIS = {
    ROOT / "ctis_mask_october2020_daily.csv": ("smoothed_wwearing_mask", "20201001-20201031"),
    ROOT / "vaccine_experiment/acceptance_2021-02.csv": (
        "smoothed_waccept_covid_vaccine", "20210201-20210228"),
}


def main():
    provenance = {}
    for path, (signal, dates) in CTIS.items():
        url = ("https://api.delphi.cmu.edu/epidata/covidcast/?data_source=fb-survey"
               f"&signal={signal}&time_type=day&geo_type=county&time_values={dates}"
               "&geo_value=*&format=csv")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            download(url, path)
        frame = pd.read_csv(path, dtype={"geo_value": str})
        if frame.empty or not frame.signal.eq(signal).all():
            raise ValueError(f"Invalid CTIS response in {path}; check the CSV and API availability.")
        provenance[path.as_posix()] = {"url": url,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if signal == "smoothed_wwearing_mask":
            monthly, audit = ctis_monthly(frame)
            out = Path("data/transformed/ctis_october2020_monthly_proxy.csv")
            out.parent.mkdir(parents=True, exist_ok=True)
            monthly.to_csv(out, index=False)
            provenance[path.as_posix()]["aggregation"] = audit
    (ROOT / "weekly_ctis_sources.json").write_text(json.dumps(provenance, indent=2))
    print("Weekly CTIS inputs ready; hashes in data/raw/weekly_ctis_sources.json")


if __name__ == "__main__":
    main()
