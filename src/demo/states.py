from typing_extensions import TypedDict

class InputState(TypedDict):
    user_input: str

class WorkingState(TypedDict):
    rag_result: str

class OutputState(TypedDict):
    graph_output: str


