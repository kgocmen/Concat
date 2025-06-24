import os
import ast
import pandas as pd
from collections import Counter


class DatasetProcessor:
    def __init__(self, input_dir="./data", output_dir="./data", seed=42):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.seed = seed
        os.makedirs(self.output_dir, exist_ok=True)
        self.NECESSARY_KEYS = [ "amenity", "addr", "leisure", "name", "tourism", "place", "shop", "building"]

    def load_csv(self, filename):
        return pd.read_csv(filename)

    def save_csv(self, df, filename):
        df.to_csv(filename, index=False)
        print(f"✅ Saved to {filename}")

    def sample_dataset(self, input_csv, output_csv, sample_sizes):
        df = self.load_csv(input_csv)
        for n in sample_sizes:
            sample_n = n * 1000
            sampled_df = df.sample(n=sample_n, random_state=self.seed)
            self.save_csv(sampled_df, output_csv)

    def clean_by_necessary_keys(self, input_csv, output_csv):
        df = self.load_csv(input_csv)
        df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else {})

        def has_necessary_key(tags):
            return any(k.split(":")[0] in self.NECESSARY_KEYS for k in tags)

        cleaned_df = df[df["tags"].apply(has_necessary_key)]
        self.save_csv(cleaned_df, output_csv)
        print(f"✅ Cleaned dataset with {len(cleaned_df)} rows saved to '{output_csv}'.")
