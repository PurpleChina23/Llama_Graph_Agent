"""
Memory management for the LLM Agent

This module provides memory functionality for storing and retrieving conversation history.
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from datetime import datetime
import json


class ConversationMemory:
    """Manages conversation memory for the LLM agent."""
    
    def __init__(self, max_messages: int = 100):
        """Initialize conversation memory.
        
        Args:
            max_messages: Maximum number of messages to keep in memory
        """
        self.max_messages = max_messages
        self.messages: List[BaseMessage] = []
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the conversation memory.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        # Trim messages if exceeding max limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_human_message(self, content: str) -> None:
        """Add a human message to the conversation.
        
        Args:
            content: The message content
        """
        message = HumanMessage(content=content)
        self.add_message(message)
    
    def add_ai_message(self, content: str) -> None:
        """Add an AI message to the conversation.
        
        Args:
            content: The message content
        """
        message = AIMessage(content=content)
        self.add_message(message)
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation.
        
        Args:
            content: The message content
        """
        message = SystemMessage(content=content)
        self.add_message(message)
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation.
        
        Returns:
            List of all messages
        """
        return self.messages.copy()
    
    def get_recent_messages(self, n: int = 10) -> List[BaseMessage]:
        """Get the most recent n messages.
        
        Args:
            n: Number of recent messages to return
            
        Returns:
            List of recent messages
        """
        return self.messages[-n:] if len(self.messages) >= n else self.messages
    
    def clear_memory(self) -> None:
        """Clear all messages from memory."""
        self.messages.clear()
        self.metadata.clear()
        print("✅ Conversation memory cleared")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation.
        
        Returns:
            Dictionary containing conversation summary
        """
        human_count = sum(1 for msg in self.messages if isinstance(msg, HumanMessage))
        ai_count = sum(1 for msg in self.messages if isinstance(msg, AIMessage))
        system_count = sum(1 for msg in self.messages if isinstance(msg, SystemMessage))
        
        return {
            "total_messages": len(self.messages),
            "human_messages": human_count,
            "ai_messages": ai_count,
            "system_messages": system_count,
            "memory_usage": f"{len(self.messages)}/{self.max_messages}",
            "metadata": self.metadata
        }
    
    def save_to_file(self, filepath: str) -> bool:
        """Save conversation memory to a file.
        
        Args:
            filepath: Path to save the memory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            data = {
                "messages": [msg.dict() for msg in self.messages],
                "metadata": self.metadata,
                "saved_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Conversation memory saved to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to save memory: {e}")
            return False
    
    def load_from_file(self, filepath: str) -> bool:
        """Load conversation memory from a file.
        
        Args:
            filepath: Path to load the memory from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Reconstruct messages
            self.messages = []
            for msg_data in data.get("messages", []):
                msg_type = msg_data.get("type", "")
                content = msg_data.get("content", "")
                
                if msg_type == "human":
                    self.messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    self.messages.append(AIMessage(content=content))
                elif msg_type == "system":
                    self.messages.append(SystemMessage(content=content))
            
            self.metadata = data.get("metadata", {})
            print(f"✅ Conversation memory loaded from {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to load memory: {e}")
            return False


class LongTermMemory:
    """Manages long-term memory for the agent."""
    
    def __init__(self, storage_path: str = "agent_memory.json"):
        """Initialize long-term memory.
        
        Args:
            storage_path: Path to store long-term memory
        """
        self.storage_path = storage_path
        self.memories: Dict[str, Any] = {}
        self.load_memory()
    
    def store_memory(self, key: str, value: Any) -> None:
        """Store a memory with a key.
        
        Args:
            key: Memory key
            value: Memory value
        """
        self.memories[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat()
        }
        self.save_memory()
    
    def retrieve_memory(self, key: str) -> Optional[Any]:
        """Retrieve a memory by key.
        
        Args:
            key: Memory key
            
        Returns:
            Memory value if found, None otherwise
        """
        if key in self.memories:
            return self.memories[key]["value"]
        return None
    
    def get_all_memories(self) -> Dict[str, Any]:
        """Get all stored memories.
        
        Returns:
            Dictionary of all memories
        """
        return self.memories.copy()
    
    def delete_memory(self, key: str) -> bool:
        """Delete a memory by key.
        
        Args:
            key: Memory key to delete
            
        Returns:
            True if deleted, False if not found
        """
        if key in self.memories:
            del self.memories[key]
            self.save_memory()
            return True
        return False
    
    def save_memory(self) -> bool:
        """Save memory to file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"❌ Failed to save long-term memory: {e}")
            return False
    
    def load_memory(self) -> bool:
        """Load memory from file.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                self.memories = json.load(f)
            return True
        except FileNotFoundError:
            print(f"⚠️ Memory file {self.storage_path} not found. Starting with empty memory.")
            return True
        except Exception as e:
            print(f"❌ Failed to load long-term memory: {e}")
            return False


# Global memory instances
conversation_memory = ConversationMemory()
long_term_memory = LongTermMemory()


def get_conversation_memory() -> ConversationMemory:
    """Get the global conversation memory instance.
    
    Returns:
        The global ConversationMemory instance
    """
    return conversation_memory


def get_long_term_memory() -> LongTermMemory:
    """Get the global long-term memory instance.
    
    Returns:
        The global LongTermMemory instance
    """
    return long_term_memory