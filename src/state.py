from langgraph.graph import MessagesState
from typing import List, Any, Type, Optional
from pydantic import BaseModel


class State(MessagesState):
    query: str
    cur_round_num: int  # number of agent-calling rounds taken so far
    max_round_num: int  # maximum number of agent-calling rounds before a forced final answer

    final_answer_structure: Optional[Type[BaseModel]]
    field_by_field: bool

    # ReAct trajectory: one entry per agent-calling step, each with
    # "thought", "action", "agent_instructions" and (once the agents return) "observation".
    react_steps: List[Any]
    reasoning_result: Any  # latest step (thought + action)
    decision_result: Any   # {"thought", "final_answer", "is_final"}; is_final is False when the round budget forced the answer

    text_summary: Any
    image_summary: Any
    video_summary: Any
    audio_summary: Any

    text_agent_result: Any
    image_agent_result: Any
    video_agent_result: Any
    audio_agent_result: Any

    text_agent_result_list: Any
    image_agent_result_list: Any
    video_agent_result_list: Any
    audio_agent_result_list: Any

    text: List[Any]
    image: List[Any]
    video: List[Any]
    audio: List[Any]
