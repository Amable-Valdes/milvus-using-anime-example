import pandas as pd
import numpy as np
from pymilvus import MilvusClient, DataType

# Load CSVs
df = pd.read_csv("./dataset/embeddings_df.csv", index_col=0)
print(f"Loaded dataset with shape: {df.shape}")
df_names = pd.read_csv("./dataset/65k_anime_data.csv")
print(f"Loaded dataset with shape: {df_names.shape}")

# Get the data
vector_cols = df.columns.tolist()
vectors = df[vector_cols].values.tolist()
ids = df.index.tolist()
names = df_names["title"].to_list()

# Connect to Milvus
print("Connecting...")
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("Conection stablished\n")

# Select database to use
database_name = "anime_database"
client.use_database(db_name=database_name)

# Insert new series that didnt exist before (new vectors for the DB)
print("Inserting data...")
collection_name = "anime_embeddings_collection"
data = [
    {
        "anime_index": ids[2790], 
        "anime_name": str(names[2790]),
        "embedding": vectors[2790]
    },
    {
        "anime_index": ids[1896], 
        "anime_name": str(names[1896]),
        "embedding": vectors[1896]
    },
    {
        "anime_index": ids[1344], 
        "anime_name": str(names[1344]),
        "embedding": vectors[1344]
    }
]

print(f"Example data that is going to be inserted:")
print(f"\tanime_index: {data[0]["anime_index"]}")
print(f"\tanime_name: {data[0]["anime_name"]}")
print(f"\tembedding: {np.asarray(data[0]["embedding"], dtype=np.float32)}")
res = client.insert(
    collection_name=collection_name,
    data=data
)
print(f"Inserted {res["insert_count"]} rows into Milvus collection '{collection_name}'\n")

client.flush(collection_name=collection_name)
