from .state import State
from langgraph.graph import StateGraph, START, END
from .nodes import react_reason, react_observe, summaries_ready, ACTION_CALL_AGENTS
from .subagents.text_agent import text_agent, text_summarize
from .subagents.image_agent import image_agent, image_summarize
from .subagents.video_agent import video_agent, video_summarize
from .subagents.audio_agent import audio_agent, audio_summarize
from .config import config

SUBAGENT_NAMES = config["agents"]["names"]


def route_action(state: State):
    """Act: run only the agents selected in the latest step, or stop on a final answer."""
    step = state["reasoning_result"]
    if step["action"] != ACTION_CALL_AGENTS:
        return END
    return [instruction["agent_name"] for instruction in step["agent_instructions"]]


def _build_graph():
    builder = StateGraph(State)

    summarize_nodes = [text_summarize, image_summarize, video_summarize, audio_summarize]
    agent_nodes = [text_agent, image_agent, video_agent, audio_agent]

    for node in [*summarize_nodes, *agent_nodes, summaries_ready, react_reason, react_observe]:
        builder.add_node(node)

    # Summarize every modality once, in parallel.
    for node in summarize_nodes:
        builder.add_edge(START, node.__name__)
        builder.add_edge(node.__name__, summaries_ready.__name__)
    builder.add_edge(summaries_ready.__name__, react_reason.__name__)

    # ReAct loop: reason (thought + action) -> selected agents -> observe -> reason ...
    builder.add_conditional_edges(
        react_reason.__name__,
        route_action,
        [*SUBAGENT_NAMES, END],
    )
    for agent_name in SUBAGENT_NAMES:
        builder.add_edge(agent_name, react_observe.__name__)
    builder.add_edge(react_observe.__name__, react_reason.__name__)

    return builder.compile()


_graph = _build_graph()
