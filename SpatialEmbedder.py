import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class SpatialEmbedder:
    """
    Computes spatial-only embeddings for latitude/longitude points
    relative to fixed anchors, saves normalized vectors in .npy format.

    Parameters
    ----------
    input_csv : str         – Path to input CSV with 'lat' and 'lon' columns
    output_npy : str        – Path to output .npy file
    chunk_size : int        – Number of rows to process per chunk
    anchors : list          – List of (lat, lon) anchor tuples
    """

    def __init__(self, input_csv: str, output_npy: str, chunk_size: int = 500_000):
        self.input_csv = input_csv
        self.output_npy = output_npy
        self.chunk_size = chunk_size

        os.makedirs(os.path.dirname(self.output_npy), exist_ok=True)

        self.df = pd.read_csv(self.input_csv)
        self.total_rows = len(self.df)

        self.lat_min = self.df["lat"].min()
        self.lat_max = self.df["lat"].max()
        self.lon_min = self.df["lon"].min()
        self.lon_max = self.df["lon"].max()

        self.lat_range = self.lat_max - self.lat_min
        self.lon_range = (self.lon_max - self.lon_min) * np.cos(np.radians((self.lat_min + self.lat_max) / 2))

        # Default anchors: corners of bounding box
        self.anchors = [(self.lat_min, self.lon_min), (self.lat_max, self.lon_max)]
        self.spatial_dim = len(self.anchors) * 2

    def run(self):
        print("🔧 Generating spatial-only embeddings")
        fused_array = np.lib.format.open_memmap(
            self.output_npy,
            dtype=np.float32,
            mode='w+',
            shape=(self.total_rows, self.spatial_dim)
        )

        for start in range(0, self.total_rows, self.chunk_size):
            end = min(start + self.chunk_size, self.total_rows)
            print(f"➡️  Processing rows {start} to {end}")

            for j, (_, row) in tqdm(
                enumerate(self.df.iloc[start:end].iterrows()),
                total=(end - start),
                desc=f"chunk {start // self.chunk_size + 1}",
                leave=False
            ):
                spatial_block = []
                for anchor_lat, anchor_lon in self.anchors:
                    spatial_block.extend(self._encode_relative(row["lat"], row["lon"], anchor_lat, anchor_lon))

                fused = np.array(spatial_block, dtype=np.float32)
                fused /= np.linalg.norm(fused)
                fused_array[start + j] = fused

        print(f"✅ Saved spatial-only embeddings to: {self.output_npy}")

    def _encode_relative(self, lat, lon, anchor_lat, anchor_lon):
        lat_offset = abs(lat - anchor_lat)
        lon_offset = abs(lon - anchor_lon) * np.cos(np.radians(lat))
        lat_score = 1 - (lat_offset / self.lat_range)
        lon_score = 1 - (lon_offset / self.lon_range)
        return [lat_score, lon_score]


# Example usage
if __name__ == "__main__":
    embedder = SpatialEmbedder(
        input_csv="./data/cleaned_melbourne.csv",
        output_npy="./spatial/spatial_embeddings.npy",
        chunk_size=500_000
    )
    embedder.run()
