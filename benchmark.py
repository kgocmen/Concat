import DatasetProcessor
import SemanticEmbedder
import SpatialEmbedder
import ConcatQuery
import os


# CHANGE THESE
TO_CLEAN = False # Set to True if you want to clean the dataset
TO_SAMPLE = True # Set to True if you want to sample the dataset
SAMPLE_SIZE = 100 # Sample size in thousands
INPUT_CSV = "./data/melbourne.csv" # Path to dataset
# END CHANGE



processor = DatasetProcessor.DatasetProcessor()

if INPUT_CSV.find("cleaned") == -1 and TO_CLEAN:
    new_csv = INPUT_CSV.replace(".csv", "_cleaned.csv")
    if os.path.exists(new_csv):
        print(f"✅ Skipping cleaning, '{new_csv}' already exists.")
    else:
        processor.clean_by_necessary_keys(INPUT_CSV, new_csv)
    INPUT_CSV = new_csv
if INPUT_CSV.find("sampled") == -1 and TO_SAMPLE:
    new_csv = INPUT_CSV.replace(".csv", f"_sampled_{SAMPLE_SIZE}k.csv")
    if os.path.exists(new_csv):
        print(f"✅ Skipping sampling, '{new_csv}' already exists.")
    else:
        processor.sample_dataset(INPUT_CSV, new_csv, sample_sizes=[SAMPLE_SIZE])
    INPUT_CSV = new_csv



semantic_embeddings_path = "semantic/semantic_embeddings_" + INPUT_CSV.split("/")[-1].replace(".csv", "") + ".npy"
spatial_embeddings_path = "spatial/spatial_embeddings_" + INPUT_CSV.split("/")[-1].replace(".csv", "") + ".npy"

if os.path.exists(semantic_embeddings_path):
    print("Semantic embeddings already exist, skipping semantic embedding generation.")
else:
    embedder = SemanticEmbedder.SemanticEmbedder(
        input_csv=INPUT_CSV,
        output_npy=semantic_embeddings_path,
        chunk_size=100_000,
    )
    embedder.run()

if os.path.exists(spatial_embeddings_path):
    print("Spatial embeddings already exist, skipping spatial embedding generation.")
else:
    embedder = SpatialEmbedder.SpatialEmbedder(
        input_csv=INPUT_CSV,
        output_npy=spatial_embeddings_path,
        chunk_size=500_000
    )
    embedder.run()

concat = ConcatQuery.ConcatQuery(
    csv_path=INPUT_CSV,
    semantic_path=semantic_embeddings_path,
    spatial_path=spatial_embeddings_path,
    output_path=f"output/{INPUT_CSV.split('/')[-1].replace('.csv', '')}.txt",
    lambdas=[1,2,3,4]
)

queries = [
    ("Where can I buy medicine?", -37.8000, 144.9710),
    ("Where can I buy a book?", -37.6000, 144.9000),
    ("Where can I find a good restaurant?", -37.8670, 144.9780),
    ("What are some popular tourist attractions?", -37.8136, 144.9631),
    ("Dandenong", -37.9833, 145.2167),
    ("Shrine of Remembrance", -34.0000, 148.0000),
]

concat.run_queries(queries)
print("All tasks completed successfully.")