from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

class AgentStatus(str):
    IDLE = "IDLE"
    WORKING = "WORKING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"

class Message(BaseModel):
    """Base unit of communication between agents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    recipient: str
    priority: int = Field(default=1, ge=1, le=5)  # 1 is normal, 5 is critical
    content: str
    image_url: Optional[str] = None # For multi-modal input (Vision)
    image_base64: Optional[str] = None # For screenshot data
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Task(Message):
    """A specific request for an agent to perform an action."""
    task_type: str  # e.g., "RESEARCH", "CODE", "REVIEW"
    requirements: List[str] = Field(default_factory=list)
    deadline: Optional[datetime] = None

class Observation(Message):
    """The result of an action or tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    resource_usage: Dict[str, float] = Field(default_factory=dict)  # e.g. {"tokens": 100, "time_ms": 500}

class Goal(BaseModel):
    """A high-level objective for the system."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    description: str
    status: str = "PENDING"
    subtasks: List[str] = Field(default_factory=list)  # IDs of subtasks
    created_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
