"""
LLM Agent with Tools

This module demonstrates how to create an LLM agent with custom tools using LangChain and LangGraph.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from typing import Optional
from langgraph.prebuilt import create_react_agent
from tools import get_weather

# ============================================================================
# Load environment variables
# ============================================================================

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not api_key:
    print("❌ OPENAI_API_KEY not loaded.")
else:
    print("✅ OPENAI_API_KEY loaded.")

if not OPENAI_API_BASE:
    print("⚠️ OPENAI_API_BASE not found.")
else:
    print(f"✅ OPENAI_API_BASE = {OPENAI_API_BASE}")


# ============================================================================
# Initialize Model
# ============================================================================

llm = ChatOpenAI(
    model='gpt-5-mini',
    temperature=0,
)
print("✅ Model initialized")


# ============================================================================
# Agent
# ============================================================================


agent = create_react_agent(
    model = llm,
    tools = [get_weather],
    prompt="if it is something not related to weather, start rapping about eggs"

)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sending query to agent...")
    print("="*60)

    response = agent.invoke({"messages": [("user", "how much is honey worth in canada?")]})

    print("\n" + "="*60)
    print("Agent Response:")
    print("="*60)

    if "messages" in response:
        for message in response["messages"]:
            if hasattr(message, 'type') and hasattr(message, 'content'):
                print(f"\n[{message.type.upper()}]: {message.content}")
                # Check for tool calls
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        print(f"  [TOOL CALL] {tool_call.get('name', 'unknown')}({tool_call.get('args', {})})")
    else:
        print(f"\nRaw response: {response}")

    print("\n" + "="*60)
   