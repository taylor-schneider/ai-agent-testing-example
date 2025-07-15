from demo.rag_agent import RAGAgent

agent = RAGAgent()
request = {"messages": [{"role": "user", "content": "what is taylor's last name?"}]}
response = agent.invoke(request)
result = response["messages"][-1].content
s = ""