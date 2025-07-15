import os
import logging
import requests
import json
import pandas
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
import hashlib
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import redis


# Configure logging
log_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(module)s:%(funcName)s():%(lineno)d] %(message)s'
logging.basicConfig(
    format=log_format,
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

# Determine the current directory and the extract path
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)

# Load the environment variables
load_dotenv(dotenv_path=root_dir)
qdrant_api_url = os.environ["QDRANT_API_URL"]
redis_api_host = os.environ["REDIS_API_HOST"]
redis_api_port = os.environ["REDIS_API_PORT"]

# Create an local instance of the embedding model
logging.debug("Creatig local embedding model")
full_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_name = 'all-MiniLM-L6-v2'
model_path = os.path.join(root_dir, ".models", model_name)
if not os.path.exists(model_path):
    embedding_model = SentenceTransformer(full_model_name)
    embedding_model.save(model_path)
else:
    embedding_model = SentenceTransformer(model_path)

# Define the information we want to embed
chunks = [
    "Taylor is 36 years old.",
    "Taylor lives at 555 Awesome Street, Chicago IL.",
    "Taylor's credit score is 700.",
    "Taylor's prefers to ride his bike rather than sit in traffic.",
    "Taylor is from the east coast",
    "Taylor's social security number is 555-5555-5555",
    "Taylor has short brown hair.",
    "Taylor banks at Chase. His bank account number is 555 555 555 55555"
]

# Convert the text into embeddings using the huggingface model
embeddings = embedding_model.encode(chunks)

# Load embeddings into a dataframe
embeddings_df = pandas.DataFrame(embeddings)

# Make a connection to the vector database
logging.debug("Connecting to vector db")
client = QdrantClient(url=qdrant_api_url)

# Create a collection in the vectordb
# Specify the length of the vectors and the distance algorithm
collection_name = "demo_data"
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.DOT)
    )

# Upsert the embeddings
points = []
for i in range(0, embeddings.shape[0]):
    vector = embeddings[i]
    point = PointStruct(id=i, vector=vector, payload={"chunk_id": i})
    points.append(point)
    
    
operation_info = client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points,
)

# Make a connection to the keyvalue store
logging.debug("Connecting to kv store")
from redis.commands.search.field import TextField
r = redis.Redis(host=redis_api_host, port=redis_api_port)

# Upload the chunks
doc = {
    "name": "demo_data",
    "chunks": chunks
}

r.json().set(f"docs:demo_data", "$", doc)