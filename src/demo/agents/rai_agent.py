from langgraph.prebuilt import create_react_agent
from demo.tools import check_text
class RAIAgent:
    
    def __init__(self, tools=[]):
        
        default_tools=[
            
        ]   
        
        if not tools:
            tools = default_tools
        
        self.agent = create_react_agent(
            model="openai:gpt-3.5-turbo",  
            tools=tools,
            prompt="""You are a Responsible AI agent. 
You must notify us if there is any information about a person's address.
You muse return either True or False.
Return True if there is a person's address or False if the response is OK.
Only return True or False
"""
        )
        
    def invoke(self, text):
        # Use the rai tool to check if the text is ok
        request = {"messages": [{"role": "user", "content": text}]}
        response = self.agent.invoke(request)
        
        # Parse the result
        result = response["messages"][-1].content.lower()
        
        if result not in ["true", "false"]:
            raise Exception("RAI agent returned a bad output.")
        
        result = result == "true"
        
        # If it's not ok, return standard message
        if result:
            return "The text failed the RAI check. Try another query."
        # If it is ok, return the original text
        
        return text



