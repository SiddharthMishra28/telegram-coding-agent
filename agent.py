import base64
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("coding-agent")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class WriteFileInput(BaseModel):
    path: str = Field(description="Relative path within the workspace, e.g. app/main.py")
    content: str = Field(description="Full text content to write into the file")


class RunCodeInput(BaseModel):
    file_path: str = Field(description="Relative path to the Python file to execute, e.g. app/main.py")


class RateLimiter:
    def __init__(self):
        self.requests: list[float] = []
        self.lock = threading.Lock()

    def wait(self, rpm: int = 60):
        min_interval = 60.0 / rpm
        now = time.time()
        with self.lock:
            self.requests = [t for t in self.requests if now - t < 60]
            if len(self.requests) >= rpm:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.info("Rate limit reached, sleeping for %.1f seconds", sleep_time)
                    time.sleep(sleep_time)
                    self.requests = [t for t in self.requests if time.time() - t < 60]
            self.requests.append(time.time())


_rate_limiter = RateLimiter()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        base_url=os.getenv("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
    )


def create_agent(system_prompt: str, workspace_dir: str) -> Any:
    @tool(args_schema=WriteFileInput)
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file inside the workspace."""
        full_path = os.path.join(workspace_dir, path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"

    @tool
    def read_file(path: str) -> str:
        """Read the full text contents of a file from the workspace."""
        full_path = os.path.join(workspace_dir, path)
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()

    @tool(args_schema=RunCodeInput)
    def run_code(file_path: str) -> str:
        """Execute a Python file in the workspace and return its stdout and stderr."""
        full_path = os.path.join(workspace_dir, file_path)
        if not os.path.exists(full_path):
            return f"Error: {file_path} does not exist."
        result = subprocess.run(
            ["python", full_path],
            capture_output=True,
            text=True,
            cwd=workspace_dir,
        )
        output = result.stdout
        if result.stderr:
            output += "\nSTDERR:\n" + result.stderr
        if result.returncode != 0:
            output += f"\nReturn code: {result.returncode}"
        return output or "(no output)"

    @tool
    def list_files(directory: str = ".") -> str:
        """List files in the workspace directory."""
        full_path = os.path.join(workspace_dir, directory)
        try:
            entries = sorted(os.listdir(full_path))
            if not entries:
                return "(empty directory)"
            return "\n".join(entries)
        except Exception as e:
            return f"Error listing directory: {e}"

    tools = [write_file, read_file, run_code, list_files]
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)

    FINAL_INSTRUCTION = (
        "\n\nCRITICAL: You MUST end this turn with a plain-text summary message. "
        "Do NOT continue calling tools forever. After your last tool call, provide a final summary."
    )

    def agent_node(state: AgentState):
        messages = [{"role": "system", "content": system_prompt + FINAL_INSTRUCTION}] + state["messages"]
        logger.info("Invoking LLM with %d messages", len(messages))
        _rate_limiter.wait()
        response = llm_with_tools.invoke(messages)
        logger.info("LLM response content='%s' tool_calls=%s", response.content[:200], len(response.tool_calls or []))
        return {"messages": [response]}

    def tools_node(state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []
        tool_map = {t.name: t for t in tools}
        for call in last_message.tool_calls:
            selected_tool = tool_map[call["name"]]
            logger.info("Running tool %s with args %s", call["name"], call["args"])
            observation = selected_tool.invoke(call["args"])
            logger.info("Tool %s result: %s", call["name"], observation[:500])
            tool_messages.append(
                ToolMessage(content=str(observation), tool_call_id=call["id"])
            )
        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    return graph.compile()


def _estimate_tokens(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", "")) // 4
    return total


def _compact_messages(messages: list[BaseMessage], max_tokens: int = 60000) -> list[BaseMessage]:
    if _estimate_tokens(messages) <= max_tokens:
        return messages

    kept = []
    for m in messages:
        if isinstance(m, HumanMessage):
            kept.append(m)
        elif isinstance(m, AIMessage) and not m.tool_calls:
            kept.append(m)
        elif isinstance(m, ToolMessage):
            kept.append(m)
        else:
            continue

        if _estimate_tokens(kept) > max_tokens:
            if len(kept) > 2:
                kept.pop(0)

    if not kept:
        return [HumanMessage(content="Continue from where you left off. Provide a final summary.")]
    return kept


def run_agent_checkpointed(system_prompt: str, workspace_dir: str, messages: list, checkpoint_path: str, max_steps: int = 500) -> dict:
    agent = create_agent(system_prompt, workspace_dir)
    state = {"messages": messages}
    consecutive_tool_calls = 0

    for step in range(max_steps):
        try:
            state = agent.invoke(state, config={"recursion_limit": 100000})
        except Exception as e:
            if "rate" in str(e).lower() or "429" in str(e):
                raise
            logger.error("Agent step %d failed: %s", step, e)
            _save_checkpoint(checkpoint_path, state, step, str(e))
            raise

        _save_checkpoint(checkpoint_path, state, step, None)

        last = state["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            return state

        if isinstance(last, AIMessage) and last.tool_calls:
            consecutive_tool_calls += 1
        else:
            consecutive_tool_calls = 0

        if consecutive_tool_calls >= 20:
            logger.warning("Forcing summary after %d consecutive tool calls", consecutive_tool_calls)
            summary_msg = HumanMessage(content="You have made many tool calls. Now provide a final plain-text summary of everything you built.")
            state["messages"].append(summary_msg)
            try:
                state = agent.invoke(state, config={"recursion_limit": 100000})
                _save_checkpoint(checkpoint_path, state, step + 1, None)
            except Exception as e:
                if "rate" in str(e).lower() or "429" in str(e):
                    raise
                logger.error("Summary step failed: %s", e)
                _save_checkpoint(checkpoint_path, state, step + 1, str(e))
                raise
            return state

        state["messages"] = _compact_messages(state["messages"])

    return state


def _save_checkpoint(path: str, state: dict, step: int, error: str | None):
    try:
        checkpoint = {
            "step": step,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": [
                {
                    "type": m.__class__.__name__,
                    "content": getattr(m, "content", ""),
                    "tool_calls": getattr(m, "tool_calls", None),
                    "additional_kwargs": getattr(m, "additional_kwargs", {}),
                }
                for m in state.get("messages", [])
            ],
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=2)
        import os as _os
        _os.replace(tmp, path)
    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)


def load_checkpoint(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None
