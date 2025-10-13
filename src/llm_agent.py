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
    model='gpt-5-nano',
    temperature=0,
    base_url=OPENAI_API_BASE,
    api_key=SecretStr(api_key) if api_key else None,
)
print("✅ Model initialized")


# ============================================================================
# Agent
# ============================================================================

# Load custom prompt from markdown file
with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
    system_message = f.read()

# Create agent with prompt parameter
# Note: checkpointer=False disables state persistence to avoid message ordering issues
agent = create_react_agent(
    model=llm,
    tools=[query_knowledge_base],
    prompt=system_message,
    checkpointer=False  # Disable checkpointing to prevent state corruption
)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sending query to agent...")
    print("="*60)

    # Create fresh message with proper format
    user_message = (
        "- Driver Swing Speed: Average 121.5 mph, Peak 124.4 mph\n"
        "- Ball Speed: Up to 180 mph, notable 186 mph\n"
        "- Height: 6 feet 1 inch (185 cm)\n"
        "- Weight: 185 pounds (84 kg)\n"
        "- Age: 49 years old\n"
        "- What should my driver be like?"
    )

    response = agent.invoke({"messages": [("user", user_message)]})

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
   