#!/usr/bin/env python3
# streamlit_app.py
#
# Streamlit UI for the Deep Research agent.
# Features:
# - Search config toggle: Tavily / OpenAI
# - Real-time tool call logs (with reflections)
# - Notes & raw notes display
# - Final report preview
#
# Run:
#   streamlit run streamlit_app.py
#
# Requirements (in your env):
#   - streamlit
#   - langgraph, langchain, langchain-core
#   - tavily (if using Tavily)
#   - openai-compatible LLMs as configured in your project
#
# This app assumes your repository structure contains the `open_deep_research`
# package with modules: configuration, deep_researcher, state, prompts, utils, etc.

import os
import asyncio
import json
from uuid import uuid4
from typing import Any, Dict, List, Optional

import streamlit as st

# Import graph & config schema from your package
from langchain_core.messages import BaseMessage, HumanMessage

from open_deep_research.deep_researcher import deep_researcher
from open_deep_research.configuration import SearchAPI

# ---------- JSON Serialization Helper ----------

def safe_json(data):
    """Recursively convert LangChain messages and other objects into JSON-serializable forms."""
    if isinstance(data, BaseMessage):
        return {"type": data.type, "content": data.content}
    elif isinstance(data, dict):
        return {k: safe_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [safe_json(v) for v in data]
    else:
        # Fallback: turn unknown objects into string
        if isinstance(data, (str, int, float, bool)) or data is None:
            return data
        return str(data)

# ---------- UI Helpers ----------

def init_session_state():
    if "events" not in st.session_state:
        st.session_state.events = []
    if "final_report" not in st.session_state:
        st.session_state.final_report = None
    if "raw_notes" not in st.session_state:
        st.session_state.raw_notes = []
    if "notes" not in st.session_state:
        st.session_state.notes = []
    if "last_config" not in st.session_state:
        st.session_state.last_config = {}
    if "run_in_progress" not in st.session_state:
        st.session_state.run_in_progress = False

def search_api_from_label(label: str) -> SearchAPI:
    lookup = {
        "Tavily": SearchAPI.TAVILY,
        "OpenAI": SearchAPI.OPENAI,
        "None": SearchAPI.NONE,
    }
    return lookup.get(label, SearchAPI.TAVILY)

def build_config(
    research_model: str,
    compression_model: str,
    final_report_model: str,
    allow_clarification: bool,
    search_api_label: str,
    max_concurrent_research_units: int,
    max_researcher_iterations: int,
    max_react_tool_calls: int,
    summarization_model: str,
    max_content_length: int,
    api_keys: Dict[str, Optional[str]],
) -> Dict[str, Any]:
    """
    Build the RunnableConfig dict that LangGraph nodes expect.
    Configuration.from_runnable_config(...) inside your code will read this.
    """
    
    # Let utils pull keys from config instead of process env
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "true"

    # Minimal unique ids for store lookups, etc.
    thread_id = st.session_state.get("thread_id") or str(uuid4())
    st.session_state.thread_id = thread_id

    config: Dict[str, Any] = {
        "configurable": {
            # Keys used by your utils.get_api_key_for_model / get_tavily_api_key
            "apiKeys": {
                "OPENAI_API_KEY": api_keys.get("OPENAI_API_KEY"),
                "TAVILY_API_KEY": api_keys.get("TAVILY_API_KEY"),
            },
            # Graph config schema fields (Configuration)
            "research_model": research_model,
            "compression_model": compression_model,
            "final_report_model": final_report_model,
            "allow_clarification": allow_clarification,
            "search_api": search_api_from_label(search_api_label),
            "max_concurrent_research_units": max_concurrent_research_units,
            "max_researcher_iterations": max_researcher_iterations,
            "max_react_tool_calls": max_react_tool_calls,
            "summarization_model": summarization_model,
            "max_content_length": max_content_length,
            # Optional MCP stuff could be added here if you use it:
            # "mcp_config": {"url": "...", "auth_required": False, "tools": ["..."]},
            "thread_id": thread_id,
        },
        # Owner metadata is used by token store in utils.get_tokens()
        "metadata": {"owner": f"user-{thread_id}"},
        # Tags to avoid streaming in LangSmith (mirrors your code's tags)
        "tags": ["langsmith:nostream"],
    }
    
    return config

async def run_graph_async(user_prompt: str, config: Dict[str, Any]):
    """
    Push a user prompt through the graph and stream events in real-time.
    """
    
    st.session_state.events.clear()
    st.session_state.final_report = None
    st.session_state.raw_notes = []
    st.session_state.notes = []

    # Stream events (LangGraph v2) for real-time updates
    try:
        async for ev in deep_researcher.astream_events(
            input={"messages": [HumanMessage(content=user_prompt)]},
            config=config,
            version="v2",
        ):
            st.session_state.events.append(ev)

            # Pull state snapshots for certain nodes
            # Useful keys: ev["event"], ev["name"], ev["data"]
            if ev.get("event") == "on_chain_end" and ev.get("name") == "final_report_generation":
                # Final state will include "final_report" per your code
                output = ev.get("data", {}).get("output", {})
                if isinstance(output, dict) and "final_report" in output:
                    st.session_state.final_report = output["final_report"]
                # Raw notes/notes could be present earlier too; we'll try to capture them
            elif ev.get("event") == "on_chain_end" and ev.get("name") == "research_supervisor":
                # Supervisor subgraph end flush
                output = ev.get("data", {}).get("output", {})
                # Not strictly needed here
                pass
            elif ev.get("event") == "on_tool_end":
                # Tool output → can include think reflections, search results, etc.
                pass
    
    except Exception as e:
        st.session_state.events.append({
            "event": "error",
            "name": "exception",
            "data": {"error": str(e)},
        })

def render_event(ev: Dict[str, Any]):
    """Pretty-print a single LangGraph event in the UI."""
    
    etype = ev.get("event")
    name = ev.get("name")
    data = ev.get("data", {})

    # Group certain events
    if etype == "on_tool_start":
        with st.chat_message("assistant"):
            st.markdown(f"**Tool start**: `{name}`")
            # Show tool args, if present
            if "input" in data:
                st.code(json.dumps(safe_json(data["input"]), indent=2), language="json")

    elif etype == "on_tool_end":
        with st.chat_message("assistant"):
            st.markdown(f"**Tool end**: `{name}`")
            output = data.get("output")
            # Show a snippet for large outputs
            if isinstance(output, str) and len(output) > 2000:
                st.code(output[:2000] + "\n...[truncated]...", language="markdown")
            else:
                if isinstance(output, (dict, list)):
                    st.code(json.dumps(safe_json(output), indent=2), language="json")
                else:
                    st.code(str(output), language="markdown")

            # Highlight reflections from think_tool
            if name == "think_tool" and isinstance(output, str):
                st.info(output)

    elif etype == "on_chain_start":
        with st.chat_message("assistant"):
            st.markdown(f"**Node start**: `{name}`")

    elif etype == "on_chain_end":
        with st.chat_message("assistant"):
            st.markdown(f"**Node end**: `{name}`")
            output = data.get("output")
            if output:
                if isinstance(output, (dict, list)):
                    st.code(json.dumps(safe_json(output), indent=2), language="json")
                else:
                    st.code(str(output), language="markdown")
                
    elif etype == "on_chat_model_stream":
        # Token stream (model streaming partial outputs)
        chunk = data.get("chunk")
        if chunk:
            if hasattr(chunk, "content"):  # AIMessageChunk or similar
                content = chunk.content
            elif isinstance(chunk, dict):
                content = chunk.get("content")
            else:
                content = str(chunk)

            if content:
                with st.chat_message("assistant"):
                    st.write(content)

    elif etype == "error":
        st.error(f"Error: {data.get('error')}")

    else:
        # Fallback generic renderer
        with st.chat_message("assistant"):
            st.markdown(f"`{etype}`: **{name}**")
            if data:
                st.code(json.dumps(safe_json(data), indent=2), language="json")

def extract_notes_from_events(events: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Try to extract notes/raw_notes/tool outputs from event data for display."""
    
    raw_notes: List[str] = []
    notes: List[str] = []

    for ev in events:
        if ev.get("event") == "on_tool_end":
            name = ev.get("name")
            output = ev.get("data", {}).get("output")

            # Researcher's web_search and think_tool outputs are valuable "raw notes"
            if name in ("web_search", "tavily_search", "think_tool") and output:
                raw_notes.append(str(output))

        # You can add more heuristics here to pull state snapshots out of chain ends

    return {"raw_notes": raw_notes, "notes": notes}

# ---------- Streamlit App ----------

st.set_page_config(page_title="Deep Research – Control Panel", layout="wide")
init_session_state()

st.title("🧪 Deep Research – Control Panel")

with st.sidebar:
    st.subheader("🔐 API Keys")
    OPENAI_API_KEY = st.text_input("OpenAI API Key", type="password")
    TAVILY_API_KEY = st.text_input("Tavily API Key", type="password")
    st.caption("Keys are used only in this session and passed directly to the agent.")

    st.subheader("⚙️ Models & Limits")
    research_model = st.text_input("Research model", value="openai:o4-mini")
    compression_model = st.text_input("Compression model", value="openai:gpt-4o-mini")
    summarization_model = st.text_input("Summarization model", value="openai:gpt-4o-mini")
    final_report_model = st.text_input("Final report model", value="openai:o4-mini")
    allow_clarification = st.checkbox("Allow clarification step", value=True)

    search_api_label = st.selectbox("Search provider", ["Tavily", "OpenAI"], index=0)
    max_concurrent = st.number_input("Max concurrent research units", min_value=1, max_value=8, value=2)
    max_iterations = st.number_input("Max researcher iterations", min_value=1, max_value=15, value=6)
    max_tool_calls = st.number_input("Max ReAct tool calls (per researcher)", min_value=1, max_value=10, value=5)
    max_content_length = st.number_input("Max characters to summarize per page", min_value=2000, max_value=200000, value=60000, step=1000)

    st.markdown("---")
    st.caption("Tip: Use Tavily for current events; OpenAI search is a preview feature and may vary by account.")

# Chat Input
st.subheader("💬 Ask a research question")
prompt = st.text_area("Your prompt", placeholder="e.g., What is neuromorphic engineering? What are its past and current situations? What is its prospect? Please explain in details!", height=120)

cols = st.columns([1,1])
with cols[0]:
    run_btn = st.button("Run Deep Research", type="primary", disabled=st.session_state.run_in_progress)
with cols[1]:
    clear_btn = st.button("Clear Session")

if clear_btn:
    st.session_state["events"] = []
    st.session_state["raw_notes"] = []
    st.session_state["notes"] = []
    st.session_state["final_report"] = None
    st.session_state["run_in_progress"] = False

    # Optional: reset thread_id if needed
    st.session_state["thread_id"] = str(uuid4())

    # Also reset numeric inputs safely in session_state (optional)
    for key, default in {
        "max_concurrent": 2,
        "max_iterations": 6,
        "max_tool_calls": 5,
        "max_content_length": 60000
    }.items():
        st.session_state[key] = default
        
    max_concurrent = st.number_input(
        "Max concurrent research units",
        min_value=1, max_value=8,
        value=st.session_state.get("max_concurrent", 2),
        key="max_concurrent"
    )
    
    max_iterations = st.number_input(
        "Max researcher iterations",
        min_value=1, max_value=8,
        value=st.session_state.get("max_iterations", 2),
        key="max_iterations"
    )
    
    max_tool_calls = st.number_input(
        "Max ReAct tool calls (per researcher)",
        min_value=1, max_value=10,
        value=st.session_state.get("max_tool_calls", 2),
        key="max_tool_calls"
    )
    
    max_content_length = st.number_input(
        "Max characters to summarize per page",
        min_value=2000, max_value=200000, step=1000,
        value=st.session_state.get("max_content_length", 2),
        key="max_content_length"
    )
    
    st.rerun()

if run_btn and prompt.strip():
    st.session_state.run_in_progress = True
    config = build_config(
        research_model=research_model,
        compression_model=compression_model,
        final_report_model=final_report_model,
        allow_clarification=allow_clarification,
        search_api_label=search_api_label,
        max_concurrent_research_units=int(max_concurrent),
        max_researcher_iterations=int(max_iterations),
        max_react_tool_calls=int(max_tool_calls),
        summarization_model=summarization_model,
        max_content_length=int(max_content_length),
        api_keys={
            "OPENAI_API_KEY": OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"),
            "TAVILY_API_KEY": TAVILY_API_KEY or os.getenv("TAVILY_API_KEY"),
        },
    )

    # Run the async graph
    with st.status("Running Deep Research...", expanded=True) as status:
        asyncio.run(run_graph_async(prompt, config))
        status.update(label="Run complete", state="complete")

    # Extract notes from the captured event stream
    notes_out = extract_notes_from_events(st.session_state.events)
    st.session_state.raw_notes = notes_out["raw_notes"]
    st.session_state.notes = notes_out["notes"]

    st.session_state.run_in_progress = False

# ---------- Live Events / Tool Calls ----------

st.subheader("🔧 Tool calls (live log)")
event_container = st.container()
with event_container:
    for ev in st.session_state.events:
        # Only render events of these types
        if ev.get("event") in ("on_chain_start", "on_chain_end"):
            render_event(ev)

# ---------- Notes & Reflections ----------

st.subheader("🗒️ Notes & Reflections")
if st.session_state.raw_notes:
    with st.expander("Raw tool outputs & reflections", expanded=False):
        for i, chunk in enumerate(st.session_state.raw_notes, start=1):
            st.markdown(f"**Chunk {i}**")
            st.code(chunk, language="markdown")
else:
    st.caption("No notes captured yet. Run a query to see search results and reflections here.")

# ---------- Final Report Preview ----------

st.subheader("📄 Final Report Preview")
if st.session_state.final_report:
    st.markdown(st.session_state.final_report)
else:
    st.caption("The final report will appear here after a run completes.")