from pymilvus import MilvusClient
import pandas as pd

# Load CSVs
df = pd.read_csv("./dataset/embeddings_df.csv", index_col=0)
print(f"Loaded dataset with shape: {df.shape}")
df_names = pd.read_csv("./dataset/65k_anime_data.csv")
print(f"Loaded dataset with shape: {df_names.shape}")

vector_cols = df.columns.tolist()
vectors = df[vector_cols].values.tolist()
ids = df.index.tolist()
names = df_names["title"].to_list()

print("Connecting...")
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("Conection stablished\n")

# Select database to use
database_name = "anime_database"
client.use_database(db_name=database_name)

# Do the search
collection_name = "anime_embeddings_collection"
my_favorite_anime = vectors[9]
res = client.search(
    collection_name=collection_name,
    anns_field="embedding",
    data=[my_favorite_anime],
    limit=11,
    search_params={"metric_type": "COSINE"},
    output_fields=["anime_name"]
)

for hits in res:
    for hit in hits:
        print(hit)