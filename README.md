# Recomendation system with Milvus

In this previous work (https://www.kaggle.com/code/amablevaldes/anime-recommendation-system-based-on-embeddings) we have implemented a feature extractor that allow us to generate embeddings for our anime dataset. With these embeddings we can now generate a recommendation system using cosine similarity.

But, why stop there? Why not improve it further?

Using Milvus, a vectorDB, we can create a database, insert our embeddings into collections and run queries to find vectors similar to other vectors, natively having a vector space on which to perform operations.

# Setup Milvus

I have left you a `docker-compose.yml`on this repository. It has all the information necessary for the deployment of Milvus through docker.

If you don't have Docker installed you can install it (2025/08: https://docs.docker.com/get-started/get-docker/)

To create your Milvus DB you only need to execute `docker compose up` and the system will start. I like to use `docker compose up -d` to detach the execution and put it on background.

(2025/07: https://milvus.io/docs/install_standalone-docker.md)

# Requirements

I have left you a requirements.txt generate an environment to use this code. You can install it, but you will only need `pymilvus`, `numpy` and `pandas` to execute this code.

# Using Milvus for your embeddings

Well, first of all let's import the libs that we will use.


```python
import pandas as pd
import numpy as np
from pymilvus import MilvusClient, DataType
```

We have the dataset from the previous experiment. Let's load it on memory


```python
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
```

    Loaded dataset with shape: (29153, 480)
    Loaded dataset with shape: (29153, 19)


## Milvus client connection

Excelent. Now let's connect to the Milvus database you must have deployed. You can visit it on `http://localhost:9091/webui`. To connect a client to it, you will execute the next code:


```python
# Connect to Milvus
print("Connecting...")
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)
print("Conection stablished")
```

    Connecting...
    Conection stablished


Milvus is like any other database; you will need a client where you are going to do the operations and transactions.

## Create collections

Like any other database, Milvus has Databases (where users can operate), Tables (Called _Collections_ on Milvus, like on MongoDB) and that Tables have an Schema (combination of columns with specific data types)

Let's generate our Database where we will create our collection:

¡¡¡ATTENTION!!! This code will drop previously databases and its data


```python
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
print("Database created")
```

    Creating database
    Database created


Now that we have a Database we must define the Schema out Collection is going to have:


```python
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

print("Schema defined")
```

    Defining Collection schema...
    Schema defined


As you can see our collection is going to have 3 fields: anime_index, anime_name and embedding. The index and the name will be to identify the anime and give it a name, they are attributes. Meanwhile, the embedding is the information which we are going to work on our searchs on the vector space.

The last step is generate the Collection that will follow the precious Schema on our new created Database:


```python
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
```

    Create the Collection...
    Collection creation done:
    {'state': <LoadState: Loaded>}


After this we have an operative Collection where we can save embeddings/vectors.

## Populate Database

Now we need to populate the database with the dataset we have.

First of all, we must prepare the data as a JSON with the schema that we have defined.


```python
# Prepare the data
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

# Show results
print(f"Example data that is going to be inserted:")
print(f"\tanime_index: {data[0]["anime_index"]}")
print(f"\tanime_name: {data[0]["anime_name"]}")
print(f"\tembedding: {np.asarray(data[0]["embedding"], dtype=np.float32)}")

```

    Inserting data...
    Example data that is going to be inserted:
    	anime_index: 0
    	anime_name: Cowboy Bebop
    	embedding: [ 2.60000000e+01  0.00000000e+00  0.00000000e+00  1.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  9.97850239e-01  9.98534322e-01  9.50904250e-01
      9.17996526e-01  9.25776005e-01  9.28852797e-01 -1.02758870e-01
     -1.86214726e-02 -2.19760966e-02  4.48753941e-04  7.14786053e-02
      8.35761726e-02  5.64495027e-02  6.30009081e-03 -1.09310886e-02
     -2.67969500e-02  6.61202520e-02  7.19554815e-03 -6.01062402e-02
      2.27974616e-02 -6.76087141e-02  1.37209222e-02  2.70903837e-02
     -1.02712639e-01  3.01422551e-02  1.44898170e-03 -6.72823116e-02
     -2.85908878e-02  3.37406434e-02 -3.73854190e-02 -7.81231374e-02
      7.51347616e-02  5.35386726e-02  2.98556383e-03 -4.21745554e-02
      8.64654500e-03  8.68059322e-02  6.68808222e-02 -8.61402377e-02
      7.35497624e-02  2.10886505e-02 -4.50174175e-02  6.00210354e-02
     -2.52399389e-02  6.91841915e-02 -5.81195531e-03 -1.56507846e-02
     -2.75761001e-02 -3.66028138e-02  4.33001556e-02 -2.05806922e-02
     -1.09261088e-01 -3.20409313e-02 -4.28141616e-02  3.90020228e-04
     -4.17277217e-02 -7.20472485e-02 -5.48618846e-02  3.11948825e-02
      3.20672952e-02  8.47892687e-02  5.20165777e-03  5.86541332e-02
      7.31024742e-02  5.17951697e-02 -1.20299250e-01  2.45824680e-02
      3.92189138e-02  1.13895359e-02 -2.09365431e-02 -9.11723264e-03
     -4.55186404e-02  8.89507495e-03  7.28628710e-02  5.74998744e-02
      6.58162534e-02  2.01375037e-02  7.21146390e-02 -1.85468514e-02
      4.94841067e-03  4.72827069e-03  4.94954064e-02  2.58293934e-02
      2.15329789e-02  4.06294912e-02 -1.05890580e-01 -1.23539567e-02
     -7.01336935e-02 -8.63197222e-02 -1.72598064e-02  2.19104234e-02
     -9.05497931e-03 -7.93445185e-02 -2.00213585e-02 -3.01943887e-02
      2.06031427e-02 -2.87304930e-02 -6.29503699e-03  5.71965277e-02
     -6.48020580e-02  3.83855338e-04  4.95569743e-02 -8.56822133e-02
      5.34535088e-02 -5.70161454e-02  2.94121075e-02 -2.83789821e-02
     -2.41004732e-02  1.32241799e-02  1.13023967e-02  8.31732824e-02
     -5.81147429e-03  8.55522528e-02  3.05658877e-02 -1.44985819e-03
     -6.71408921e-02 -1.17013361e-02 -4.33316641e-02  1.53487381e-02
      4.51058000e-02  9.70020220e-02  1.08228950e-02 -2.69288626e-02
      2.12383028e-02 -4.00603078e-02 -5.28363027e-02  1.01936288e-01
      2.46673059e-02 -3.93867120e-02  1.79804061e-02 -1.13378681e-01
     -1.25644896e-02 -3.79168168e-02  5.32669461e-33  4.86808345e-02
     -1.29239792e-02 -7.21563175e-02 -1.52007267e-02 -1.43371876e-02
      3.88781391e-02 -1.11802757e-01  3.70288566e-02 -4.42938842e-02
      7.37738758e-02 -1.54600143e-01 -1.10318944e-01 -9.66965128e-03
      4.81397584e-02 -4.57543693e-02  9.43916943e-03 -4.55981307e-02
     -8.98684375e-03  1.54673625e-02  1.36549501e-02 -1.99158629e-03
      8.63081664e-02 -7.29634762e-02 -9.12586078e-02 -2.42855996e-02
     -2.93661235e-03 -5.20223193e-02 -4.02400363e-03  7.26355389e-02
      1.31731108e-02 -9.55880154e-03  1.67086661e-01  3.35109746e-03
      2.09168531e-04 -2.04303786e-02 -8.81558191e-03  2.06720475e-02
     -6.45945072e-02 -4.53736149e-02  3.30197439e-02  1.45101845e-02
      5.41407242e-02 -8.02016333e-02 -1.24674635e-02  7.89780263e-03
      8.89445022e-02  2.33661733e-03 -8.31610784e-02  1.06875701e-02
      1.98674183e-02 -4.18141531e-03 -1.90786291e-02  8.61850102e-03
     -1.28823090e-02  2.25945594e-04  8.69925097e-02  1.54084312e-02
      3.61849740e-02  8.87044892e-02  2.39572953e-02  6.79307058e-02
      3.36118899e-02  1.80072933e-02  1.61685180e-02  2.76964549e-02
     -1.51265254e-02  9.96586233e-02  3.99355553e-02 -1.67524572e-02
     -3.29905637e-02 -5.73323965e-02  2.38637645e-02 -5.20332083e-02
     -9.65715498e-02 -5.03094085e-02 -7.16882646e-02  1.04294583e-01
      1.39991585e-02 -5.72367795e-02 -3.52192745e-02 -1.21221274e-01
     -6.26088157e-02  7.02297017e-02 -1.43436994e-02 -3.89025807e-02
     -2.14542896e-02  1.40468134e-02 -7.63836950e-02  2.21455954e-02
      5.98815680e-02 -2.00958643e-02 -5.02760448e-02  3.24069336e-02
      4.01088037e-03 -1.03100114e-01 -5.92841364e-33 -4.73359879e-03
     -3.61534469e-02 -4.69110236e-02 -4.39093402e-03  3.68592143e-02
      2.61883382e-02 -2.78371088e-02  7.43510528e-03  3.66865238e-03
     -3.40099260e-02 -8.85596797e-02 -4.09935229e-02  2.49714945e-02
     -2.63303798e-02  1.19069986e-01 -5.05693257e-02 -8.77372734e-03
      3.48009840e-02 -3.15728076e-02  2.92033274e-02  7.45382905e-02
      2.55217087e-02 -7.25497752e-02 -6.15642965e-03  4.15152721e-02
      9.08482149e-02  4.05893624e-02  3.25183347e-02 -7.25784227e-02
      6.95742741e-02  2.57691853e-02  4.10718918e-02 -1.42814601e-02
     -2.68290136e-02 -7.81978481e-03  4.42421101e-02  1.40541047e-02
     -3.62300873e-02 -5.18334918e-02 -5.02077118e-02 -2.70579173e-03
      4.68358472e-02 -7.41329268e-02  8.40146020e-02 -1.05214365e-01
      2.57572811e-02 -1.43681839e-03  1.48520857e-01 -6.68585673e-02
      2.08223495e-03 -6.99551702e-02  2.29241122e-02 -1.67204160e-02
      3.92091833e-03 -1.75526626e-02 -1.75507739e-02  3.00612208e-02
     -2.82448716e-02  4.92019169e-02  7.17642903e-02 -5.40142246e-02
     -3.82315740e-02 -5.45104826e-03 -3.09652067e-03  2.36958042e-02
     -4.45664637e-02  5.13248555e-02  7.12712994e-03 -8.73783231e-02
     -1.47645371e-02  8.55368972e-02 -5.18196002e-02 -1.72281519e-01
      1.97602455e-02 -1.32103488e-02 -1.99776851e-02 -4.44378927e-02
      1.96445994e-02 -9.14511178e-03  1.37678618e-02 -1.67684145e-02
     -3.21344472e-02  2.95045916e-02  9.68841165e-02 -2.95390328e-03
      3.81516963e-02  1.68166608e-02  3.58955339e-02  5.57741150e-02
     -3.59218009e-02  3.56580305e-04 -9.87050980e-02 -1.09636979e-02
     -3.29902060e-02 -3.17068808e-02 -6.91110600e-08 -2.41744500e-02
      4.10265997e-02  8.48536845e-03  6.73989858e-03 -5.87065518e-03
      6.46833144e-03 -4.29074094e-02  5.38993580e-03 -1.07082799e-02
      8.51451010e-02 -1.78488307e-02 -5.76714501e-02 -2.70814914e-03
     -2.96138898e-02  3.26222032e-02  3.77838761e-02 -1.05684241e-02
     -5.08801304e-02 -5.48063852e-02  6.08675964e-02 -1.53720425e-02
      4.80927229e-02  1.01766713e-01 -3.39077078e-02 -1.44483531e-02
      2.83764172e-02 -1.18539091e-02 -2.65495386e-02  9.34811980e-02
      2.73217633e-02 -5.85463792e-02  7.44450092e-02  1.94413457e-02
      2.07298715e-02  5.39071113e-03 -2.81967297e-02  5.53254522e-02
      7.15973228e-02 -2.57391669e-02  3.34486850e-02  2.19806712e-02
      1.09314909e-02  3.98075953e-02 -1.85766816e-02 -1.25347218e-02
     -6.54282123e-02  6.48495108e-02 -4.45079617e-02  4.74121273e-02
      1.60484184e-02 -3.55249457e-03 -2.76908539e-02 -3.89351510e-02
      4.66873720e-02 -1.66324563e-02 -4.85183187e-02 -5.47019951e-02
      1.05436988e-01 -1.00949993e-02 -2.61972863e-02  1.22111142e-01
     -2.10897848e-02 -9.59095825e-03 -3.17259580e-02  1.00000000e+00]


We have prepared the data to be inserted.

For every sample of the dataset, we will do an insert on the database. We can send several samples at the same time as a transaction (insert_many in any database).

Note that I have deleted from the dataset prepared data 4 specific samples. We will see this later, for now ignore it.

Let's insert the data:


```python
# Insert data
res = client.insert(
    collection_name=collection_name,
    data=data
)
print(f"Inserted {res["insert_count"]} rows into Milvus collection '{collection_name}'")

client.flush(collection_name=collection_name)
```

    Inserted 29149 rows into Milvus collection 'anime_embeddings_collection'


# Search on our spatial vector space

Now that we have data on our collection, we can ask for similar vectors to other vectors. This is how recommendation systems works.

We are going o take _Cowboy Bebop_ as an example. This is the element with index 0 on our dataset, so, we ask for similar vectors and...


```python
# Do the search
query_anime_index=9
my_favorite_anime = vectors[9]
res = client.search(
    collection_name=collection_name,
    anns_field="embedding",
    data=[my_favorite_anime],
    limit=11,
    search_params={"metric_type": "COSINE"},
    output_fields=["anime_name"]
)

print(f"Query Anime: {names[query_anime_index]} (index={query_anime_index})\n")
for hits in res:
    for prediction in hits:
            print(f"- {prediction['entity']['anime_name']} (similarity={prediction['distance']:.4f})")
```

    Query Anime: Monster (index=9)
    
    - Warau Salesman (similarity=0.9991)
    - Lupin III: Part II (similarity=0.9991)
    - Maison Ikkoku (similarity=0.9990)
    - Kindaichi Shounen no Jikenbo (similarity=0.9990)
    - Eyeshield 21 (similarity=0.9990)
    - Kirarin☆Revolution (similarity=0.9990)
    - Tetsuwan Atom (similarity=0.9989)
    - Tennis no Oujisama (similarity=0.9989)
    - D.Gray-man (similarity=0.9989)
    - Ranma ½ (similarity=0.9989)
    - Gekitou! Crush Gear Turbo (similarity=0.9989)


... and we have a selection of animes that have a similar vector on our spatial vector space similar to the vector of _Cowboy Bebop_

## Continuous data integration

But, think that we have an Anime Web Page where we show animes to users. We have new animes every season, so, there will be a continuous integrations of new data.

In a real scenario, you would need to train you model again with the new data to recommend your new products. But not with an cluster oriented solution.

Let's insert new data to the collection:


```python
# Insert new series that didnt exist before (new vectors for the DB)
print("Inserting data...")
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
```

    Inserting data...
    Example data that is going to be inserted:
    	anime_index: 2790
    	anime_name: Warau Salesman
    	embedding: [ 1.03000000e+02  4.00000000e+00  0.00000000e+00  1.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
      0.00000000e+00  0.00000000e+00  1.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
      1.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  1.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
      0.00000000e+00  7.70891488e-01  7.34680355e-01  5.88371933e-01
      3.23501676e-01  6.73414290e-01  5.07734299e-01  7.30868289e-03
      1.53694237e-02 -1.30903651e-03 -3.41285728e-02  1.39952563e-02
     -5.92669025e-02  1.34021834e-01  8.91919062e-03 -3.33230547e-03
     -7.05259219e-02  4.43495214e-02  6.45447895e-02 -2.77248751e-02
      7.83958361e-02 -1.27630066e-02 -5.12713827e-02  6.40840009e-02
     -1.98342018e-02 -6.78645028e-03  3.65705267e-02  2.13457663e-02
     -5.36334179e-02  3.35364081e-02 -1.35545433e-02 -2.82243509e-02
     -4.59337085e-02 -5.80458855e-03  3.86767425e-02 -8.04215763e-03
     -2.50789709e-02 -3.28993471e-03  1.44686177e-01 -6.03317050e-03
      1.08669261e-02  5.91367185e-02  8.41520503e-02 -5.45366085e-04
     -9.81528498e-03 -2.99778171e-02  2.64194161e-02  1.49073424e-02
      7.87270442e-03 -1.24564609e-02 -4.98381779e-02  2.63521429e-02
     -2.54228543e-02  5.57406582e-02  4.00052825e-03  1.03027217e-01
      1.19277658e-02 -1.27038941e-01 -6.01757271e-03 -2.26648171e-02
      2.82084011e-02  1.29357120e-02 -2.20584478e-02  3.46146785e-02
     -2.24531237e-02 -2.96936999e-03 -3.62252817e-02 -2.06647627e-02
     -7.70146586e-03 -2.00885953e-03  8.76781791e-02  4.71441485e-02
     -4.65455465e-02 -5.35225049e-02  4.14194465e-02 -1.12906344e-01
      3.39888595e-02  4.14131843e-02 -1.62727814e-02  7.04400018e-02
     -4.10183258e-02 -8.29612240e-02  3.78584154e-02  6.16506748e-02
     -1.00484736e-01 -2.74519380e-02 -9.73667111e-03 -7.45696947e-03
     -8.44734162e-02 -5.06419735e-03 -3.93755548e-02 -7.36161508e-03
     -5.82509376e-02  6.98942784e-03 -1.09372497e-01  6.39736876e-02
     -3.90117802e-02 -2.92980950e-02 -2.63401656e-03  7.33393580e-02
     -5.97244315e-03 -8.02993029e-02  5.10874167e-02  9.70026478e-03
      1.65557805e-02 -6.16761707e-02  8.72412622e-02 -2.03760527e-02
     -9.14166272e-02  5.93620129e-02 -3.72047983e-02  2.04897355e-02
     -2.57010907e-02  3.18767056e-02 -1.04659507e-02 -1.75706006e-03
      6.17595650e-02 -1.29715621e-01 -1.25313345e-02  7.85716325e-02
     -1.13747306e-02  4.35751677e-02 -1.74312834e-02  2.62506865e-02
      9.58897099e-02 -9.10616145e-02  4.08794098e-02  7.62675330e-02
      4.66806628e-03  1.12275342e-02  4.76036295e-02 -6.65757060e-02
      2.50683632e-02 -8.80567171e-03  2.72920952e-33 -2.66218726e-02
     -1.45577686e-02  1.58380270e-02 -4.02651131e-02  9.48139951e-02
      4.20005480e-03 -3.91078591e-02 -5.07572256e-02 -2.27738880e-02
      4.13908586e-02 -9.20810476e-02 -1.20925775e-03 -6.83996677e-02
      2.78105382e-02 -2.45487946e-03  2.34674066e-02 -6.58191368e-02
      6.33565523e-03  4.27736901e-02  3.70065235e-02  5.46717830e-02
      6.31503686e-02  2.12104488e-02 -2.01619361e-02 -4.69712391e-02
      2.94022821e-02  1.38756745e-02  1.17857400e-02 -1.27008613e-02
      2.50205863e-02 -5.47229461e-02  1.26051441e-01  2.28743106e-02
     -9.65890661e-03 -3.62923220e-02  1.27284387e-02 -1.34788275e-01
     -5.03577180e-02 -3.56979705e-02 -4.69516739e-02 -1.08691998e-01
     -5.15556000e-02 -6.44023344e-02  9.01701152e-02 -3.79880667e-02
      4.45727743e-02  1.11856684e-02 -8.82384926e-03  7.92384520e-02
      3.89439017e-02 -3.53519320e-02 -5.18904924e-02 -1.72204934e-02
      1.78811373e-04 -4.83688898e-02 -9.40118283e-02 -5.25637418e-02
     -1.08328305e-01  5.11156097e-02 -5.52996621e-03 -2.84825787e-02
     -2.71556638e-02  6.32782886e-03 -2.70133391e-02 -8.71766880e-02
     -7.10098222e-02  2.91954223e-02 -2.13367287e-02  1.00605162e-02
     -7.22424174e-03 -4.83880611e-03 -5.22997323e-03 -7.61938989e-02
      3.20161991e-02  2.31438931e-02 -7.15737343e-02  3.24105322e-02
     -2.73536611e-02  6.63594762e-03  1.97878247e-03  5.07217981e-02
     -5.94854392e-02  3.58365364e-02  1.92835163e-02 -1.71420779e-02
      5.98849282e-02  5.52870147e-03 -1.39941489e-02  4.99655083e-02
      5.67156076e-02 -9.96300485e-03 -8.27570707e-02  5.21469340e-02
      3.68200019e-02  6.20774440e-02 -5.17817421e-33 -4.30377834e-02
     -2.91626398e-02 -2.81493906e-02  5.49313659e-03  1.05716297e-02
     -2.43402366e-02 -9.40711647e-02 -1.28341205e-02  4.50117849e-02
      2.61036772e-02 -1.49296701e-01 -8.52665827e-02  2.52599530e-02
      5.16876811e-03 -3.24923620e-02 -3.48420925e-02  1.37171343e-01
     -3.77873238e-03 -2.31810641e-02 -4.86165695e-02  5.66944703e-02
      2.72748768e-02 -1.12705089e-01 -2.39392389e-02 -2.68056244e-02
      4.59124781e-02  3.34183834e-02 -2.81057693e-02 -1.08505927e-01
      4.43960540e-02 -3.55578884e-02 -4.44313772e-02 -8.14726874e-02
      3.08523569e-02  4.21049967e-02 -5.15221385e-03 -7.24839941e-02
     -5.28238947e-04  8.81819148e-03  2.54952144e-02  1.16776451e-01
     -3.30809243e-02  5.70229888e-02  6.25673905e-02 -2.41340697e-02
     -4.86507341e-02 -1.06626861e-02  1.03475628e-02  5.62998280e-02
     -1.52078234e-02 -6.66523129e-02  5.66723123e-02  3.79104610e-03
     -5.75447567e-02 -8.01533461e-03  9.37232655e-03 -9.32603516e-03
      3.64670567e-02 -2.23892890e-02  2.05357056e-02  2.00205054e-02
     -7.70289153e-02  2.28186455e-02 -1.71399508e-02  6.81694970e-03
      6.68297485e-02  1.38253629e-01 -7.16966242e-02  6.31866381e-02
     -1.79730970e-02 -5.56010455e-02 -3.09090912e-02 -6.76038712e-02
     -2.97149103e-02  2.75948197e-02  6.80608600e-02 -1.44313455e-01
      6.09705858e-02 -1.70784816e-02 -5.59621751e-02  4.18377258e-02
     -4.35258634e-02  2.11075488e-02  7.41752535e-02 -1.55924158e-02
     -9.26210731e-03  1.18915280e-02  2.12417003e-02  6.19537756e-02
      9.59992991e-04  4.02833074e-02  1.57838725e-02  2.76213512e-02
     -3.65618691e-02 -2.33477093e-02 -5.00754744e-08 -2.45302003e-02
     -7.70601034e-02  4.91695292e-02 -5.96319139e-02  4.87470329e-02
      5.42000122e-02 -6.17651083e-02  1.87597293e-02 -1.44944163e-02
      1.16024710e-01  9.10399258e-02  1.50844246e-01 -6.75553875e-03
      4.93926229e-03  6.94766920e-03  7.99446180e-02  8.02922025e-02
      3.12538631e-02 -1.86889712e-02  3.55845131e-02  3.09429248e-03
      4.35098028e-03  3.89930084e-02  4.13080193e-02  1.83997191e-02
      1.64578669e-02 -6.92755803e-02  4.12705094e-02  1.57013983e-02
      5.56533970e-02 -5.10703726e-03 -4.96520568e-03 -7.94935748e-02
      8.61639855e-05 -2.18575690e-02 -5.44309791e-04 -7.45545775e-02
     -2.85339281e-02 -5.61566874e-02 -2.77032107e-02  4.29954454e-02
      5.23611233e-02  9.69660059e-02  5.50915077e-02 -4.05390374e-02
     -6.22632680e-03 -1.78310089e-02 -5.47455736e-02  4.94908467e-02
      1.93572491e-02 -6.98571792e-03 -7.02864258e-03 -2.36769971e-02
      3.61965038e-02 -1.24049876e-02 -1.47948086e-01 -7.71548459e-03
     -3.25881280e-02 -6.41207909e-04 -4.65424359e-02 -2.07241736e-02
     -1.11030834e-02  1.36076016e-02 -6.11949712e-02  1.00000000e+00]



```python
res = client.insert(
    collection_name=collection_name,
    data=data
)
print(f"Inserted {res["insert_count"]} rows into Milvus collection '{collection_name}'\n")

client.flush(collection_name=collection_name)
```

    Inserted 3 rows into Milvus collection 'anime_embeddings_collection'
    


And now let's ask again the database for similar vectors to Cowboy Bebop vector:


```python
# Do the search
query_anime_index=9
my_favorite_anime = vectors[9]
res = client.search(
    collection_name=collection_name,
    anns_field="embedding",
    data=[my_favorite_anime],
    limit=11,
    search_params={"metric_type": "COSINE"},
    output_fields=["anime_name"]
)

print(f"Query Anime: {names[query_anime_index]} (index={query_anime_index})\n")
for hits in res:
    for prediction in hits:
            print(f"- {prediction['entity']['anime_name']} (similarity={prediction['distance']:.4f})")
```

    Query Anime: Monster (index=9)
    
    - Warau Salesman (similarity=0.9991)
    - Lupin III: Part II (similarity=0.9991)
    - Maison Ikkoku (similarity=0.9990)
    - Kindaichi Shounen no Jikenbo (similarity=0.9990)
    - Eyeshield 21 (similarity=0.9990)
    - Kirarin☆Revolution (similarity=0.9990)
    - Tetsuwan Atom (similarity=0.9989)
    - Tennis no Oujisama (similarity=0.9989)
    - D.Gray-man (similarity=0.9989)
    - Ranma ½ (similarity=0.9989)
    - Gekitou! Crush Gear Turbo (similarity=0.9989)


The vectorial space has changed, and with a simple search we have an update of the data. This allows a continuous integration of the solution with other tools like kafka or a custom pipeline that feed the model with new data.

# Conclusions

Deploy a Milvus database is easy with docker.

The code implementation is like any other database.

In other cases, we would need a re-training of the model to fit new data, but with this approach our model is our infraestructure and we have _AI as Infraestructure_ with native operations of search and data persistence. 

Thanks!

Amable Valdés
