from langgraph.prebuilt import create_react_agent

import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import redis



class RAGAgent:
    
    def __init__(self, tools=[]):
        

        # Determine the current directory
        current_dir = os.path.abspath(os.path.dirname(__file__))
        self.root_dir = os.path.dirname(current_dir)

        # Load the environment variables
        load_dotenv(self.root_dir)
        self.qdrant_api_url = os.environ["QDRANT_API_URL"]
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.redis_api_host = os.environ["REDIS_API_HOST"]
        self.redis_api_port = os.environ["REDIS_API_PORT"]
        
        # Define the embedding collection and connect to the vector store
        self.vector_db_collection_name = "demo_data"
        self.vector_db_client = QdrantClient(url=self.qdrant_api_url)
        
        # Connect to the chunk store
        self.redis_client = redis.Redis(host=self.redis_api_host, port=self.redis_api_port)

        # Create an local instance of the embedding model
        full_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = 'all-MiniLM-L6-v2'
        model_path = os.path.join(self.root_dir, ".models", model_name)
        if not os.path.exists(model_path):
            self.embedding_model = SentenceTransformer(full_model_name)
            self.embedding_model.save(model_path)
        else:
            self.embedding_model = SentenceTransformer(model_path)
        
        # Define the internal agent
        tools = [self.get_user_info]
        
        self.agent = create_react_agent(
            model="openai:gpt-3.5-turbo",  
            tools=tools,
            prompt="You are a RAG agent. You lookup information in a private datastore. You expect a query or set of keywords to be supplied. Lookup the information relevant to the query."  
        )
    def get_user_info(self, query: str) -> list:
        """Search an internal data store given a general query or question.
        
        query: A string specifying a question or set of keywords to lookup
        
        """
        query_vector = self.generate_embedding(query)
        relevant_chunks = self.get_chunks_related_to_query(query_vector=query_vector, max_chunks=3)
        return relevant_chunks

    def generate_embedding(self, text):
            embedding = self.embedding_model.encode(text)
            return embedding

    def get_chunks_related_to_query(self, query_vector, max_chunks):

        # Search the vector db for the three closest matches
        search_results = self.vector_db_client.search(
            collection_name= self.vector_db_collection_name, 
            query_vector=query_vector, 
            limit=max_chunks
        )

        best_match_chuck_ids = [(search_result.payload["chunk_id"], search_result.score) for search_result in search_results]

        chunks = []
        for best_match_chuck_id in best_match_chuck_ids:
            chunk_id = best_match_chuck_id[0]
            chunk = self.redis_client.json().get(f"docs:demo_data")["chunks"][chunk_id]
            chunks.append(chunk)
        
        return chunks

    def invoke(self, query):
        request = {"messages": [{"role": "user", "content": query}]}
        response = self.agent.invoke(request)
        result = response["messages"][-1].content
        return result



