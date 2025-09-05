import pandas as pd

DATA_PATH = "./data/initial_cohort.csv"

def load_initial_cohort(path=DATA_PATH):
    """Load the initial cohort subject IDs."""
    return pd.read_csv(path)

if __name__ == "__main__":
    cohort = load_initial_cohort()
    print(f"Loaded {len(cohort)} patients from initial cohort.")
    print(cohort.head())

