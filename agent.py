import os
import subprocess
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class WriteFileInput(BaseModel):
    path: str = Field(description="Relative path within the workspace, e.g. app/main.py")
    content: str = Field(description="Full text content to write into the file")


class RunCodeInput(BaseModel):
    file_path: str = Field(description="Relative path to the Python file to execute, e.g. app/main.py")


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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def tools_node(state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []
        tool_map = {t.name: t for t in tools}
        for call in last_message.tool_calls:
            selected_tool = tool_map[call["name"]]
            observation = selected_tool.invoke(call["args"])
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
