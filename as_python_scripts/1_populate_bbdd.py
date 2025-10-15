# http://localhost:9001/login
# user and pass: minioadmin
# http://localhost:9091/webui
# user and pass: root:Milvus

import pandas as pd
import numpy as np
from pymilvus import MilvusClient, DataType

# Load CSVs
df = pd.read_csv("./dataset/embeddings_df.csv", index_col=0)
print(f"Loaded dataset with shape: {df.shape}")
df_names = pd.read_csv("./dataset/65k_anime_data.csv")
print(f"Loaded dataset with shape: {df_names.shape}")

# All columns are part of the embedding
vector_cols = df.columns.tolist()

# Convert rows of vector columns into list of floats
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

# Create database
print("Creating database")
database_name = "anime_database"
# Drop database if exists (for fresh start)
if database_name in client.list_databases():
    client.use_database(db_name=database_name)
    for collection in client.list_collections():
        client.drop_collection(collection_name=collection)
    client.drop_database(db_name=database_name)
# Create database
client.create_database(db_name=database_name)
# Select as database in use
client.use_database(db_name="anime_database")
print("Database created\n")


# Define the DB schema
print("Defining Collection schema...")

# Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)
# Add fields to schema
schema.add_field(field_name="anime_index", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="anime_name", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=len(vector_cols))
# Prepare index parameters
index_params = client.prepare_index_params()
# Add indexes
index_params.add_index(
    field_name="anime_index",
    index_type="AUTOINDEX"
)
index_params.add_index(
    field_name="anime_name",
    index_type="AUTOINDEX"
)
index_params.add_index(
    field_name="embedding", 
    index_type="AUTOINDEX",
    metric_type="COSINE"
)
print("Schema defined\n")

# Create collection
print("Create the Collection...")

# Drop collection if exists (for fresh start)
collection_name = "anime_embeddings_collection"
# Collection creation
client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params
)
res = client.get_load_state(collection_name=collection_name)
print("Collection creation done:")
print(res)
print("\n")

# Insert data
print("Inserting data...")
data = [
    {
        "anime_index": ids[i], 
        "anime_name": str(names[i]),
        "embedding": vectors[i]
    } for i in range(len(ids))
]
# Before insert the elements, i take out the example samples that i will show later
data.pop(9)
data.pop(2790)
data.pop(1896)
data.pop(1344)
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
