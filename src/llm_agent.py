"""
LLM Agent with Tools

This module demonstrates how to create an LLM agent with custom tools using LangChain and LangGraph.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
from langgraph.prebuilt import create_react_agent
from tools import query_knowledge_base
from pydantic import SecretStr

# ============================================================================
# IMPROVEMENT NEEDED: Add imports for memory and logging (2025 best practices)
# ============================================================================
# TODO: Add these imports:
# from langgraph.checkpoint.memory import MemorySaver  # For development
# from langgraph.checkpoint.postgres import PostgresSaver  # For production
# import logging
# from datetime import datetime
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(f'logs/agent_{datetime.now().strftime("%Y%m%d")}.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
# os.makedirs('logs', exist_ok=True)
# ============================================================================

# ============================================================================
# Load environment variables
# ============================================================================

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# ============================================================================
# IMPROVEMENT NEEDED: Replace print with logging
# ============================================================================
# TODO: Replace print statements with logger.info() for better production logging
# logger.info("OPENAI_API_KEY loaded successfully")
# logger.warning("OPENAI_API_BASE not found")
# ============================================================================

if not api_key:
    print("❌ OPENAI_API_KEY not loaded.")
    # TODO: Consider raising an error here instead of continuing
    # raise ValueError("OPENAI_API_KEY is required but not found")
else:
    print("✅ OPENAI_API_KEY loaded.")

if not OPENAI_API_BASE:
    print("⚠️ OPENAI_API_BASE not found.")
else:
    print(f"✅ OPENAI_API_BASE = {OPENAI_API_BASE}")


# ============================================================================
# Initialize Model
# ============================================================================

# ============================================================================
# IMPROVEMENT NEEDED: Add error handling and configuration
# ============================================================================
# TODO: Add error handling for model initialization
# TODO: Add timeout parameter (timeout=30.0)
# TODO: Consider moving config to separate config file
# ============================================================================

llm = ChatOpenAI(
    model='gpt-5-nano',
    temperature=0,  # Good: 0 for consistent, deterministic responses
    base_url=OPENAI_API_BASE,
    api_key=SecretStr(api_key) if api_key else None,
    # TODO: Add timeout=30.0 to prevent hanging
)   
print("✅ Model initialized")
# TODO: Replace with logger.info("Model initialized: gpt-5-nano")


# ============================================================================
# Agent
# ============================================================================

# ============================================================================
# IMPROVEMENT NEEDED: Add error handling for file read
# ============================================================================
# TODO: Add try-except for file reading
# try:
#     with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
#         system_message = f.read()
# except FileNotFoundError:
#     logger.error("Prompt file not found")
#     raise
# ============================================================================

# Load custom prompt from markdown file
with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
    system_message = f.read()

# ============================================================================
# 🔴 CRITICAL LIMITATION: Memory disabled!
# ============================================================================
# checkpointer=False means NO conversation memory!
#
# This causes:
# - Agent forgets previous messages in same conversation
# - Can't do multi-turn conversations
# - No error recovery (can't resume from failures)
# - Missing human-in-the-loop capabilities
# - No state inspection for debugging
#
# The comment says "to avoid message ordering issues" but this is likely
# a misunderstanding. Checkpointing is STABLE in LangGraph 2025.
#
# BENEFITS of enabling checkpointing (2025 research):
# ✅ Conversation memory across turns
# ✅ Error recovery (restart from last checkpoint)
# ✅ Human-in-the-loop workflows
# ✅ State inspection for debugging
# ✅ Fault tolerance
#
# TODO: Enable checkpointing:
# 1. Import: from langgraph.checkpoint.memory import MemorySaver
# 2. Create: checkpointer = MemorySaver()
# 3. Use: checkpointer=checkpointer (replace False)
# 4. When invoking, add config with thread_id:
#    config = {"configurable": {"thread_id": "session_001"}}
#    response = agent.invoke({"messages": [...]}, config=config)
#
# For production, use PostgresSaver instead of MemorySaver
# ============================================================================

# Create agent with prompt parameter
# Note: checkpointer=False disables state persistence to avoid message ordering issues
agent = create_react_agent(
    model=llm,
    tools=[query_knowledge_base],  # TODO: Add retrieve_knowledge_base tool when fixed
    prompt=system_message,
    checkpointer=False  # 🔴 CRITICAL: Change to MemorySaver() for conversation memory
    # TODO: Add state_modifier for better control (2025 feature)
    # TODO: Add max_iterations=10 to prevent infinite loops
)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # ============================================================================
    # IMPROVEMENT NEEDED: Add comprehensive error handling (2025 best practice)
    # ============================================================================
    # TODO: Wrap entire execution in try-except block
    # TODO: Add logging instead of print statements
    # TODO: Log execution time and token usage
    # ============================================================================

    print("\n" + "="*60)
    print("Sending query to agent...")
    print("="*60)
    # TODO: Replace with logger.info("Starting agent query")

    # Create fresh message with proper format
    user_message = (
        "- Driver Swing Speed: Average 121.5 mph, Peak 124.4 mph\n"
        "- Ball Speed: Up to 180 mph, notable 186 mph\n"
        "- Height: 6 feet 1 inch (185 cm)\n"
        "- Weight: 185 pounds (84 kg)\n"
        "- Age: 49 years old\n"
        "- What should my driver be like?"
    )

    # ============================================================================
    # IMPROVEMENT NEEDED: Add config with thread_id for memory
    # ============================================================================
    # When checkpointing is enabled, MUST pass config with thread_id:
    # config = {"configurable": {"thread_id": "session_001"}}
    # response = agent.invoke({"messages": [("user", user_message)]}, config=config)
    #
    # This enables conversation memory across multiple invoke() calls
    # ============================================================================

    response = agent.invoke({"messages": [("user", user_message)]})
    # TODO: Add config parameter when checkpointing is enabled

    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Better response parsing and error handling
    # ============================================================================
    # TODO: Add validation that response is not None
    # TODO: Add error handling for malformed responses
    # TODO: Log tool calls for debugging
    # ============================================================================

    if "messages" in response:
        for message in response["messages"]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                print(f"\n[{message.type.upper()}]: {message.content}")
                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"  [TOOL CALL] {tool_call.get('name', 'unknown')}({tool_call.get('args', {})})")
                        # TODO: Add logger.info(f"Tool called: {tool_call.get('name')}")
    else:
        print(f"\nRaw response: {response}")
        # TODO: Add logger.warning(f"Unexpected response format: {response}")

    print("\n" + "="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Add example of multi-turn conversation
    # ============================================================================
    # TODO: Show how to continue conversation with same thread_id:
    # # Follow-up question
    # follow_up = "What about the shaft flex?"
    # response2 = agent.invoke(
    #     {"messages": [("user", follow_up)]},
    #     config={"configurable": {"thread_id": "session_001"}}  # Same thread_id
    # )
    # # Agent will remember previous context!
    # ============================================================================
   