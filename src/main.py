#!/usr/bin/env python3
"""
Main entry point for the LLM Agent application.

This script provides a command-line interface for interacting with the LLM agent.
"""

import argparse
import sys
import os
from typing import Optional
from llm_agent import create_agent, LLMAgent


def print_banner():
    """Print the application banner."""
    print("=" * 60)
    print("🤖 LLM Agent with Tools")
    print("=" * 60)
    print("A powerful AI agent with custom tools and memory capabilities")
    print("=" * 60)


def print_help():
    """Print help information."""
    print("\nAvailable commands:")
    print("  chat <message>     - Chat with the agent")
    print("  tools              - List available tools")
    print("  interactive        - Start interactive mode")
    print("  test               - Run test suite")
    print("  help               - Show this help")
    print("  quit/exit          - Exit the application")


def list_tools(agent: LLMAgent):
    """List all available tools."""
    tools = agent.tools
    print(f"\n📋 Available Tools ({len(tools)}):")
    print("-" * 40)
    for i, tool in enumerate(tools, 1):
        print(f"{i:2d}. {tool.name}")


def run_tests(agent: LLMAgent):
    """Run the test suite."""
    print("\n🧪 Running Test Suite...")
    print("=" * 60)
    
    tests = [
        ("Math calculation", "What is 15 multiplied by 23?"),
        ("Weather query", "What's the weather like in Tokyo?"),
        ("Time query", "What time is it now?"),
    ]
    
    for i, (test_name, query) in enumerate(tests, 1):
        print(f"\nTest {i}: {test_name}")
        print("-" * 40)
        response = agent.chat_with_display(query)
        if "error" in response:
            print(f"❌ Test failed: {response['error']}")
    
    print("\n✅ Test suite completed!")


def interactive_mode(agent: LLMAgent):
    """Start interactive mode."""
    print("\n🎯 Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                print_help()
            elif user_input.lower() == 'tools':
                list_tools(agent)
            elif user_input.lower() == 'test':
                run_tests(agent)
            else:
                # Chat with the agent
                response = agent.chat_with_display(user_input)
                if "error" in response:
                    print(f"❌ Error: {response['error']}")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except EOFError:
            print("\n👋 Goodbye!")
            break


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="LLM Agent with Tools")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0, help="Model temperature")
    parser.add_argument("--command", help="Command to execute")
    parser.add_argument("--message", help="Message for chat command")
    
    args = parser.parse_args()
    
    print_banner()
    
    # Create agent
    print("🚀 Initializing agent...")
    agent = create_agent(model_name=args.model, temperature=args.temperature)
    
    if not agent.is_ready():
        print("❌ Agent failed to initialize. Please check your API key and try again.")
        print("💡 Make sure you have set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    print(f"✅ Agent ready with {len(agent.tools)} tools")
    
    # Handle command line arguments
    if args.command:
        if args.command == "chat" and args.message:
            response = agent.chat_with_display(args.message)
            if "error" in response:
                print(f"❌ Error: {response['error']}")
        elif args.command == "tools":
            list_tools(agent)
        elif args.command == "test":
            run_tests(agent)
        elif args.command == "interactive":
            interactive_mode(agent)
        else:
            print("❌ Invalid command or missing arguments")
            print_help()
    else:
        # Default to interactive mode
        interactive_mode(agent)


if __name__ == "__main__":
    main()