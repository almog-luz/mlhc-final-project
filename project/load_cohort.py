import pandas as pd
import os
from typing import Union

DATA_PATH = "./data/initial_cohort.csv"

def load_initial_cohort(path: Union[str, "os.PathLike[str]"] = DATA_PATH) -> pd.DataFrame:
    """Load initial cohort CSV.

    Returns DataFrame expected to include at minimum a subject_id column.
    """
    return pd.read_csv(path)

if __name__ == "__main__":
    cohort = load_initial_cohort()
    print(f"Loaded {len(cohort)} patients from initial cohort.")
    print(cohort.head())

