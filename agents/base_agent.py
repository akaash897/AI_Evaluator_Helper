# agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

class AgentResult:
    def __init__(self, success: bool, data: Any = None, error: str = None, confidence: float = 1.0):
        self.success = success
        self.data = data
        self.error = error
        self.confidence = confidence
        self.timestamp = datetime.now().isoformat()

class BaseAgent(ABC):
    def __init__(self, name: str, tools: List[str] = None):
        self.name = name
        self.tools = tools or []
        self.memory = {}
        self.execution_history = []
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> AgentResult:
        pass
    
    def log_execution(self, task: Dict, result: AgentResult):
        self.execution_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": {
                "success": result.success,
                "error": result.error,
                "confidence": result.confidence
            }
        })
    
    def get_success_rate(self) -> float:
        if not self.execution_history:
            return 1.0
        successful = sum(1 for h in self.execution_history if h["result"]["success"])
        return successful / len(self.execution_history)