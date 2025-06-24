import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize


class ConcatQuery:
    def __init__(
        self,
        csv_path,
        semantic_path,
        spatial_path,
        output_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        lambdas=(1, 2, 3, 4),
        batch_size=1_000_000
    ):
        self.csv_path = csv_path
        self.semantic_path = semantic_path
        self.spatial_path = spatial_path
        self.output_path = output_path
        self.model = SentenceTransformer(model_name)
        self.lambdas = lambdas
        self.batch_size = batch_size

        self.df = pd.read_csv(csv_path)
        self.semantic_matrix = np.load(semantic_path, mmap_mode='r')
        self.spatial_matrix = np.load(spatial_path, mmap_mode='r')

        self.lat_min = self.df["lat"].min()
        self.lat_max = self.df["lat"].max()
        self.lon_min = self.df["lon"].min()
        self.lon_max = self.df["lon"].max()
        self.lat_range = self.lat_max - self.lat_min
        self.lon_range = (self.lon_max - self.lon_min) * np.cos(np.radians((self.lat_min + self.lat_max) / 2))
        self.anchors = [(self.lat_min, self.lon_min), (self.lat_max, self.lon_max)]

    def haversine(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def encode_spatial(self, lat, lon):
        spatial = []
        for anchor_lat, anchor_lon in self.anchors:
            lat_diff = abs(lat - anchor_lat)
            lon_diff = abs(lon - anchor_lon) * np.cos(np.radians(lat))
            spatial.append(1 - (lat_diff / self.lat_range))
            spatial.append(1 - (lon_diff / self.lon_range))
        return np.array(spatial)

    def run_queries(self, queries):
        lines = []

        for query_text, qlat, qlon in queries:
            print(f"🔍 Query: '{query_text}' at ({qlat:.4f}, {qlon:.4f})")
            lines.append(f"=== Query: '{query_text}' at ({qlat:.4f}, {qlon:.4f}) ===")

            query_vec = self.model.encode(query_text)
            query_vec /= np.linalg.norm(query_vec)

            query_spatial = self.encode_spatial(qlat, qlon)
            query_spatial /= np.linalg.norm(query_spatial)

            # --- Spatial Only ---
            sims_spa = np.dot(self.spatial_matrix, query_spatial)
            lines.extend(self._report_topk(sims_spa, qlat, qlon, "-- Spatial Only --"))

            # --- Semantic Only ---
            sims_sem = np.dot(self.semantic_matrix, query_vec)
            lines.extend(self._report_topk(sims_sem, qlat, qlon, "-- Semantic Only (λ = 0) --"))

            # --- Joint (Fused) ---
            for λ in self.lambdas:
                fused_lines = self._fused_search(query_vec, query_spatial, qlat, qlon, λ)
                lines.extend(fused_lines)

        self._write_output(lines)

    def _fused_search(self, query_vec, query_spatial, qlat, qlon, λ):
        query_spatial_lambda = np.tile(query_spatial, λ)
        query_fused = np.concatenate([query_spatial_lambda, query_vec])
        query_fused /= np.linalg.norm(query_fused)

        sims_joint = []

        for i in range(0, len(self.df), self.batch_size):
            spa_chunk = self.spatial_matrix[i:i+self.batch_size]
            sem_chunk = self.semantic_matrix[i:i+self.batch_size]

            spa_lambda = np.hstack([spa_chunk] * λ)
            fused = np.hstack([spa_lambda, sem_chunk])
            fused = normalize(fused, axis=1)

            chunk_scores = np.dot(fused, query_fused)
            sims_joint.extend(chunk_scores)

        sims_joint = np.array(sims_joint)
        return self._report_topk(sims_joint, qlat, qlon, f"-- λ = {λ} --")

    def _report_topk(self, scores, qlat, qlon, header, k=10):
        topk = scores.argsort()[-k:][::-1]
        lines = [f"\n{header}"]
        for i in topk:
            row = self.df.iloc[i]
            dist = self.haversine(qlat, qlon, row["lat"], row["lon"])
            lines.append(f"Score: {scores[i]:.4f} | Distance: {dist:.2f} km | ID: {row['id']} | tags: {row['tags']}")
        print(f"{header.strip()} completed")
        return lines

    def _write_output(self, lines):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✅ Comparison results saved to '{self.output_path}'")