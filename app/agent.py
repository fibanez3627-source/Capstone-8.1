"""LangGraph agent :  (entry) -> decide -> [act -> decide]* -> answer -> END

"""
from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.config import settings
from app.errors import is_fatal_api_error
from app.prompts import ANSWER_SYSTEM, TOOL_SYSTEM, with_instructions
from app.textutils import parse_json
from app.tools import RequestContext

log = logging.getLogger("rag.agent")

_TOOL_MAIN_ARG = {"semantic_search": "query", "drawing_search": "query",
                  "datasheet_search": "query", "manual_search": "query",
                  "product_datasheet": "part",
                  "list_documents": "keyword", "read_document": "doc", "table_lookup": "code",
                  "manufacturer_web_search": "query", "web_search": "query",
                  "calculator": "expression", "clarify": "question", "answer": "text"}


def step_signature(tool: str, args: dict) -> str:
    """Same format for seeded steps and agent steps, so repeats are detected."""
    return f"[{tool} {args}] ->"


class AgentState(TypedDict):
    question: str
    transcript: list[str]
    iterations: int
    decision: dict
    answer: str
    clarify: bool


def _ctx(config: RunnableConfig) -> RequestContext:
    return config["configurable"]["ctx"]


class HybridGraphAgent:
    def __init__(self, llm: Any, max_steps: int | None = None) -> None:
        self.llm = llm
        self.max_steps = max_steps or settings.agent_max_steps
        self.app = self._build_graph()

    def choose_tool(self, ctx: RequestContext, question: str, transcript: str) -> dict:
        user = (f"Question: {question}\n\nSteps so far:\n{transcript or '(none)'}\n\n"
                "What is the next tool call?")
        raw = ctx.meter.invoke(self.llm, [SystemMessage(content=with_instructions(
                                              TOOL_SYSTEM, ctx.instructions)),
                                          HumanMessage(content=user)], stage="agent_decide")
        try:
            d = parse_json(raw)
            if not isinstance(d, dict):
                raise ValueError("not an object")
        except (json.JSONDecodeError, ValueError):
            return {"tool": "answer", "args": {}, "reasoning": "parse-fail"}
        tool = str(d.get("tool") or "answer")
        args = d.get("args") or {}
        if not isinstance(args, dict):
            args = {_TOOL_MAIN_ARG.get(tool, "query"): str(args)}
        return {"tool": tool, "args": args, "reasoning": str(d.get("reasoning", ""))}

    # ---------------- nodes ----------------
    def _decide_node(self, state: AgentState, config: RunnableConfig) -> dict:
        d = self.choose_tool(_ctx(config), state["question"], "\n".join(state["transcript"]))
        line = f"[decide] {d['tool']}({d['args']}) :: {d['reasoning'][:70]}"
        log.debug("step %d: %s", state["iterations"] + 1, line)
        return {"decision": d, "transcript": state["transcript"] + [line]}

    def _act_node(self, state: AgentState, config: RunnableConfig) -> dict:
        ctx = _ctx(config)
        d = state["decision"]
        tool, args = d["tool"], d["args"]
        sig = step_signature(tool, args)
        fn = ctx.tools.get(tool)
        if any(t.startswith(sig) for t in state["transcript"]):
            obs = "(repeated call skipped — use the notes above, change the query, or answer)"
        elif fn is None:
            obs = f"ERROR: unknown tool '{tool}'. Valid: {', '.join(ctx.tools)}, clarify, answer"
        else:
            try:
                obs = fn(**args)
            except Exception as e:
                if is_fatal_api_error(e):
                    raise
                obs = f"tool error ({tool} {args}): {e}"
        return {"transcript": state["transcript"] + [f"{sig} {obs}"],
                "iterations": state["iterations"] + 1}

    def _answer_node(self, state: AgentState, config: RunnableConfig) -> dict:
        d = state["decision"]
        args = d.get("args") or {}
        if d.get("tool") == "clarify":
            q = str(args.get("question") or args.get("text") or "").strip()
            return {"answer": q or "¿Puedes precisar un poco más tu pregunta?", "clarify": True}
        # Always write the final answer from the notes. The text the tool-picking step puts
        # in answer(text) is often a note to itself ("[1] and [2] already contain the
        # tables...") or a summary, not the complete answer the user needs.
        notes = "\n".join(state["transcript"])
        ans = _ctx(config).meter.invoke(
            self.llm, [SystemMessage(content=with_instructions(
                ANSWER_SYSTEM, _ctx(config).instructions)),
                       HumanMessage(content=f"Notes:\n{notes}\n\nQuestion: {state['question']}")],
            stage="agent_answer")
        return {"answer": ans}

    def _route(self, state: AgentState) -> str:
        tool = state["decision"]["tool"]
        if tool in ("answer", "clarify") or state["iterations"] >= self.max_steps:
            return "answer"
        return "act"

    def _route_after_act(self, state: AgentState) -> str:
        return "answer" if state["iterations"] >= self.max_steps else "decide"

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        g.add_node("decide", self._decide_node)
        g.add_node("act", self._act_node)
        g.add_node("answer", self._answer_node)
        g.set_entry_point("decide")
        g.add_conditional_edges("decide", self._route, {"act": "act", "answer": "answer"})
        g.add_conditional_edges("act", self._route_after_act, {"decide": "decide", "answer": "answer"})
        g.add_edge("answer", END)
        return g.compile()

    def run(self, question: str, ctx: RequestContext,
            seed_transcript: list[str] | None = None) -> dict:
        return self.app.invoke(
            {"question": question, "transcript": list(seed_transcript or []), "iterations": 0,
             "decision": {}, "answer": "", "clarify": False},
            config={"recursion_limit": 2 * self.max_steps + 10, "configurable": {"ctx": ctx}})
