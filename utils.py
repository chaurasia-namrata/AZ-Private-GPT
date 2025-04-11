import json
import os
from datetime import datetime
from typing import Dict, List, Optional

CONVERSATIONS_DIR = os.path.join(os.path.dirname(__file__), 'data', 'conversations')

def ensure_data_dirs():
    """Ensure the data directories exist"""
    if not os.path.exists(CONVERSATIONS_DIR):
        os.makedirs(CONVERSATIONS_DIR)

def get_conversation_path(user_id: str, conv_id: str) -> str:
    """Get the path to a conversation JSON file"""
    return os.path.join(CONVERSATIONS_DIR, f"{user_id}_{conv_id}.json")

def save_conversation(user_id: str, conv_id: str, conversation: Dict) -> None:
    """Save a conversation to JSON file"""
    ensure_data_dirs()
    file_path = get_conversation_path(user_id, conv_id)
    with open(file_path, 'w') as f:
        json.dump(conversation, f, indent=2)

def load_conversation(user_id: str, conv_id: str) -> Optional[Dict]:
    """Load a conversation from JSON file"""
    file_path = get_conversation_path(user_id, conv_id)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None

def delete_conversation(user_id: str, conv_id: str) -> bool:
    """Delete a conversation JSON file"""
    file_path = get_conversation_path(user_id, conv_id)
    if os.path.exists(file_path):
        os.remove(file_path)
        return True
    return False

def list_conversations(user_id: str) -> List[Dict]:
    """List all conversations for a user"""
    ensure_data_dirs()
    conversations = []
    prefix = f"{user_id}_"
    
    for filename in os.listdir(CONVERSATIONS_DIR):
        if filename.startswith(prefix) and filename.endswith('.json'):
            file_path = os.path.join(CONVERSATIONS_DIR, filename)
            with open(file_path, 'r') as f:
                conversation = json.load(f)
                conversations.append(conversation)
    
    # Sort by creation time, newest first
    return sorted(conversations, key=lambda x: x.get('created_at', ''), reverse=True)
