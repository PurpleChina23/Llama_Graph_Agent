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

# Load custom prompt from markdown file
with open("src/Prompt/golf_advisor_prompt.md", "r", encoding="utf-8") as f:
    system_message = f.read()

agent = create_react_agent(
    model=llm,
    tools=[query_knowledge_base],
    prompt=system_message
)

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Sending query to agent...")
    print("="*60)

    response = agent.invoke({"messages": [("user",
            "- Driver Swing Speed: Average 121.5 mph, Peak 124.4 mph (source: tglgolf.com)\n"
            "- Ball Speed: Up to 180 mph, notable 186 mph in 2007\n"
            "- Height: 6 feet 1 inch (185 cm)\n"
            "- Weight: 185 pounds (84 kg)\n"
            "- Age: 49 years oldwhat should my driver be like")]})

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
   