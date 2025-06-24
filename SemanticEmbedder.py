import os
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class SemanticEmbedder:
    """
    Stream a large CSV in chunks, create sentence-transformer embeddings
    from each row's `tags` column, and append to a single .npy file
    (with automatic resume capability).

    Parameters
    ----------
    input_csv : str  - Path to the CSV containing a `tags` column
    output_npy : str - Target file for embeddings
    model_name : str - Sentence-Transformer model identifier
    chunk_size : int - Rows per pandas chunk (default 100_000)
    """

    def __init__(
        self,
        input_csv: str,
        output_npy: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 100_000,
    ):
        self.input_csv = input_csv
        self.output_npy = output_npy
        self.chunk_size = chunk_size
        self.model = SentenceTransformer(model_name)

        # Make sure target directory exists
        os.makedirs(os.path.dirname(self.output_npy), exist_ok=True)

        # Resume support -----------------------------------------------------
        if os.path.exists(self.output_npy):
            self._embeddings = np.load(self.output_npy, mmap_mode="r")
            self.processed_rows = self._embeddings.shape[0]
            self._first_write = False
            print(f"🔄 Resuming from row {self.processed_rows}")
        else:
            self._embeddings = None
            self.processed_rows = 0
            self._first_write = True

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def run(self):
        """Start (or resume) embedding generation."""
        reader = pd.read_csv(self.input_csv, chunksize=self.chunk_size)

        current_row = 0
        total_rows = self.processed_rows

        for i, chunk in enumerate(reader, start=1):
            chunk_start = current_row
            chunk_end = current_row + len(chunk)
            current_row = chunk_end

            # Skip chunks that are already done
            if chunk_end <= self.processed_rows:
                print(f"⏩ Skipping chunk {i} (rows {chunk_start}–{chunk_end})")
                continue

            print(f"\n🔄 Processing chunk {i} (rows {chunk_start}–{chunk_end})")

            # --------------------------------------------------------------
            # 1. Parse tags column safely
            chunk["tags"] = chunk["tags"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
            )
            chunk = chunk[chunk["tags"].apply(lambda d: isinstance(d, dict) and len(d) > 0)]

            # --------------------------------------------------------------
            # 2. Embed each tag dict
            embeddings = self._embed_chunk(chunk["tags"], i)

            # --------------------------------------------------------------
            # 3. Persist to disk
            self._append_embeddings(embeddings)

            total_rows += len(embeddings)
            print(f"✅ Appended {len(embeddings)} embeddings (total written: {total_rows})")

        print(f"\n🎉 Done! All embeddings saved to '{self.output_npy}'")

    # --------------------------------------------------------------------- #
    #  Helpers
    # --------------------------------------------------------------------- #
    def _embed_chunk(self, tags_series, chunk_idx):
        """Convert a pandas Series of tag dictionaries into vectors."""
        vectors = []
        for tags in tqdm(tags_series, desc=f"Embedding chunk {chunk_idx}", leave=False):
            tag_text = "; ".join(f"{k}: {v}" for k, v in tags.items())
            vectors.append(self.model.encode(tag_text))
        return np.asarray(vectors)

    def _append_embeddings(self, new_vecs: np.ndarray):
        """Append or create the .npy file in an idempotent way."""
        if self._first_write:
            np.save(self.output_npy, new_vecs)
            self._first_write = False
        else:
            # Memory-map existing, concatenate, then overwrite
            existing = np.load(self.output_npy, mmap_mode="r")
            combined = np.concatenate((existing, new_vecs), axis=0)
            np.save(self.output_npy, combined)