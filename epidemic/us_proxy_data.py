"""Observed weekly networks for a pre-vaccination, prior-survey replay."""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .midterm_mobility import one


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(one, pd.date_range('2020-09-07', periods=12, freq='7D')):
            pass


if __name__ == '__main__':
    main()
