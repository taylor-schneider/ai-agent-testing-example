from demo.states import InputState, OutputState, WorkingState
from demo.agents.rag_agent import RAGAgent
from demo.agents.rai_agent import RAIAgent
from langgraph.graph import StateGraph
from langgraph.graph import START, END

class MyProcessGraph:
    
    def __init__(self):
    
        # Define the agents
        self.rag_agent = RAGAgent()
        self.rai_agent = RAIAgent()
        
        # Define the nodes in the graph
        self.builder = StateGraph(WorkingState, input_schema=InputState, output_schema=OutputState)
        self.builder.add_node("rag_agent", self.rag_node)
        self.builder.add_node("rai_agent", self.rai_node)
        
        # Define the edges in the graph
        self.builder.add_edge(START, "rag_agent")
        self.builder.add_edge("rag_agent", "rai_agent")
        self.builder.add_edge("rai_agent", END)
        
        # Compile the graph
        self.graph = self.builder.compile()

    def rag_node(self, state: InputState) -> WorkingState:
        query = state["user_input"]
        result = self.rag_agent.invoke(query)
        working_state_update = {"rag_result": result}
        return working_state_update

    def rai_node(self, state: WorkingState) -> OutputState:
        query = state["rag_result"]
        result = self.rai_agent.invoke(query)
        output_state_update = {"graph_output": result}
        return output_state_update
    
    def invoke(self, query: str) -> OutputState:
        input_state = {"user_input": query}
        output_state = self.graph.invoke(input_state)
        result = output_state["graph_output"]
        return result
        

