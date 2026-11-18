import json
import time
import traceback
from typing import List, Literal, Optional, Type

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, conlist, create_model

from src.models import master_model

from .config import config
from .state import State
from .utils import combine_available_agent, combine_summaries, postprocessing

ACTION_CALL_AGENTS = "call_agents"
ACTION_FINAL_ANSWER = "final_answer"


def available_agent_names(state) -> List[str]:
    """Agents whose modality is present in the user's input."""
    return [
        f"{name}_agent"
        for name in config["agents"]["supported"]
        if (state.get(f"{name}_summary") or "").strip()
    ]


def build_step_structure(state, force_final: bool) -> Type[BaseModel]:
    """Schema of one ReAct step: a thought plus exactly one action."""
    final_answer_structure: Optional[Type[BaseModel]] = state.get("final_answer_structure") or str
    final_answer_description = (
        "The answer shown to the end user. If the user's query includes output format "
        "requirements, follow them strictly."
    )
    thought = (str, Field(
        description="Your reasoning about what is known so far, what is still missing, and what to do next."
    ))

    if force_final:
        return create_model(
            "ReActFinalStep",
            thought=thought,
            action=(Literal[ACTION_FINAL_ANSWER], Field(description=f'Must be "{ACTION_FINAL_ANSWER}".')),
            final_answer=(final_answer_structure, Field(description=final_answer_description)),
        )

    class AgentInstruction(BaseModel):
        agent_name: str = Field(
            description=f"The agent to call, selected from the agent pool: {available_agent_names(state)}."
        )
        questions: conlist(str, min_length=1) = Field(
            description="Self-contained questions or instructions for this agent."
        )

    return create_model(
        "ReActStep",
        thought=thought,
        action=(Literal[ACTION_CALL_AGENTS, ACTION_FINAL_ANSWER], Field(
            description=(
                f'"{ACTION_CALL_AGENTS}" to gather more evidence from the agents, '
                f'or "{ACTION_FINAL_ANSWER}" to finish.'
            )
        )),
        agent_instructions=(Optional[List[AgentInstruction]], Field(
            default=None,
            description=f'Required when action is "{ACTION_CALL_AGENTS}", otherwise null.',
        )),
        final_answer=(Optional[final_answer_structure], Field(
            default=None,
            description=f'Required when action is "{ACTION_FINAL_ANSWER}", otherwise null. {final_answer_description}',
        )),
    )


def build_system_prompt(state, force_final: bool) -> str:
    step_num = len(state.get("react_steps") or []) + 1
    max_round_num = state.get("max_round_num", 1)

    if force_final:
        budget = (
            "You cannot call agents anymore. "
            f'You must now respond with action "{ACTION_FINAL_ANSWER}" using the evidence collected so far.'
        )
    else:
        budget = (
            f"This is step {step_num}. You may call agents in at most {max_round_num} rounds in total; "
            "after that you must give the final answer."
        )

    return f"""You are the Master Agent of the "Understanding Anything" system. This system interprets user input across multiple modalities (text, image, video, and audio) by orchestrating existing foundation models through specialized agents. The system does not rely on fine-tuning or retraining.

You work in a ReAct loop. At every step you produce a Thought and then exactly one Action:
- Thought: reason about the user's intent, what the observations so far have established, what is still missing or inconsistent, and what to do next.
- Action "{ACTION_CALL_AGENTS}": send questions to one or more agents from the Agent Pool. Their answers come back to you as an Observation, and you then take another step.
- Action "{ACTION_FINAL_ANSWER}": finish and return the answer to the user.

Rules for "{ACTION_CALL_AGENTS}":
- You cannot see the raw input yourself. You only have the short summaries below, so rely on the agents for evidence.
- Each agent has access only to the user's input in its own modality. It does not see the user's query, the other modalities, the other questions, or any earlier step.
- Therefore every question must be self-contained and explicitly include all context it needs from the user's query.
- Only call agents that can contribute. You may call several agents in one step, and ask each of them several diverse questions.
- In the first step, consider including the user's original query as one of the questions for each selected agent.
- In later steps, do not repeat questions that were already answered. Ask follow-up questions that resolve gaps, ambiguities, or contradictions in earlier observations.

Rules for "{ACTION_FINAL_ANSWER}":
- Choose it only when the observations are sufficient to answer the user's query completely and correctly. If you are not confident, gather more evidence first.
- Synthesize the observations into one coherent answer. Never repeat agent responses verbatim.
- Be strictly faithful to the user's query and intent. Do not speculate or add information the user did not ask for.
- If the user's query specifies an output format, follow it strictly.
- Do not mention agents, steps, or observations. Answer as if this were a single interaction.

Background:
1. Modality of user's input with short summaries:
{combine_summaries(state)}
2. Agent Pool:
{combine_available_agent(state)}
3. {budget}"""


def format_observation(step) -> str:
    return "Observation:\n" + json.dumps(step.get("observation"), ensure_ascii=False, indent=2, default=str)


def build_messages(state, system_prompt, format_instructions):
    """System prompt, the user's query, then the Thought/Action/Observation trajectory."""
    messages = [
        SystemMessage(content=[{"type": "text", "text": system_prompt + f"\n\n{format_instructions}"}]),
        HumanMessage(content=[{"type": "text", "text": state["query"]}]),
    ]
    for step in state.get("react_steps") or []:
        action = {
            "thought": step["thought"],
            "action": step["action"],
            "agent_instructions": step.get("agent_instructions"),
            "final_answer": None,
        }
        messages.append(AIMessage(content=json.dumps(action, ensure_ascii=False)))
        messages.append(HumanMessage(content=[{"type": "text", "text": format_observation(step)}]))
    return messages


def normalize_step(state, result: dict, force_final: bool) -> dict:
    """Validate one parsed step and merge duplicated agents. Raises ValueError to trigger a retry."""
    if force_final or result["action"] == ACTION_FINAL_ANSWER:
        if result.get("final_answer") is None:
            raise ValueError(f'action is "{ACTION_FINAL_ANSWER}" but final_answer is empty')
        return {
            "thought": result["thought"],
            "action": ACTION_FINAL_ANSWER,
            "final_answer": result["final_answer"],
        }

    available = available_agent_names(state)
    merged = {}
    for instruction in result.get("agent_instructions") or []:
        name = instruction["agent_name"]
        if name not in available:
            continue
        merged.setdefault(name, [])
        merged[name].extend(q for q in instruction["questions"] if q not in merged[name])
    if not merged:
        raise ValueError(f'action is "{ACTION_CALL_AGENTS}" but no agent from {available} was selected')

    return {
        "thought": result["thought"],
        "action": ACTION_CALL_AGENTS,
        "agent_instructions": [
            {"agent_name": name, "questions": questions} for name, questions in merged.items()
        ],
    }


def react_reason(state: State):
    """Thought + Action: decide whether to call agents again or to answer."""
    steps = state.get("react_steps") or []
    force_final = len(steps) >= state.get("max_round_num", 1) or not available_agent_names(state)

    structure = build_step_structure(state, force_final)
    parser = PydanticOutputParser(pydantic_object=structure)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=master_model.model)

    system_prompt = build_system_prompt(state, force_final)
    messages = build_messages(state, system_prompt, parser.get_format_instructions())

    retry = config["system"]["retry_times"]
    while retry:
        try:
            content = postprocessing(master_model.invoke(messages))
            try:
                result = parser.parse(content).model_dump()
            except Exception:
                result = fixing_parser.parse(content).model_dump()
            step = normalize_step(state, result, force_final)
            break
        except Exception as e:
            retry -= 1
            print(e, "remaining retry time:", retry)
            traceback.print_exc()
            if retry <= 0:
                raise e
            time.sleep(2)

    if step["action"] == ACTION_FINAL_ANSWER:
        return {
            "reasoning_result": step,
            "decision_result": {
                "thought": step["thought"],
                "final_answer": step["final_answer"],
                "is_final": not force_final,
            },
        }

    return {
        "reasoning_result": step,
        "react_steps": [*steps, step],
        "cur_round_num": len(steps) + 1,
    }


def react_observe(state: State):
    """Observation: attach the answers of the agents called in the latest step."""
    steps = list(state.get("react_steps") or [])
    step = dict(steps[-1])
    step["observation"] = {
        instruction["agent_name"]: state.get(f"{instruction['agent_name']}_result")
        for instruction in step["agent_instructions"]
    }
    steps[-1] = step
    return {"react_steps": steps}


def summaries_ready(state: State):
    """Join point after the per-modality summaries."""
    return {}
