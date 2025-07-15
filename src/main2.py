from demo.agents.rag_agent import RAGAgent
from demo.graph import MyProcessGraph


query = "Get me taylor's address and phone number."
graph = MyProcessGraph()
response = graph.invoke(query)

s = ""