#! /usr/bin/env python3

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_stratified_sample(input_file, output_file, sample_size):
    # Load the dataset into a Pandas DataFrame
    df = pd.read_json(input_file)

    # Handle cases where sample_size exceeds dataset size
    if sample_size >= len(df):
        stratified_sample = df.sample(n=len(df), random_state=42)
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=sample_size,
            # random_state=42  # For reproducibility
        )
        # Split the data while preserving the 'vul' ratio
        _, test_idx = next(sss.split(df, df["vul"]))
        stratified_sample = df.iloc[test_idx]

    stratified_sample.to_json(output_file, orient="records", indent=2)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 4:
        print("sample.py <input_file> <output_file> <sample_size>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    sample_size = int(sys.argv[3])

    create_stratified_sample(input_file, output_file, sample_size)
