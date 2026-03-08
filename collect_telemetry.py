import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class AgentMetrics:
    agent_name: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class MetricsTracker:
    def __init__(self):
        self.history: List[AgentMetrics] = []

    def record(self, agent_name: str, latency: float, usage_metadata: Any) -> None:
        """
        Enregistre les métriques d'une exécution d'agent.
        L'objet usage_metadata provient de l'API Google GenAI (GenerateContentResponse.usage_metadata).
        """
        # Extraction sécurisée des compteurs de tokens selon le schéma de l'API GenAI
        prompt_tokens = getattr(usage_metadata, "prompt_token_count", 0)
        completion_tokens = getattr(usage_metadata, "candidates_token_count", 0)
        total_tokens = getattr(usage_metadata, "total_token_count", 0)

        metrics = AgentMetrics(
            agent_name=agent_name,
            latency_seconds=round(latency, 3),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens
        )
        self.history.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """
        Agrège les données de l'ensemble du pipeline pour le reporting final.
        """
        total_latency = sum(m.latency_seconds for m in self.history)
        total_prompt = sum(m.prompt_tokens for m in self.history)
        total_completion = sum(m.completion_tokens for m in self.history)

        return {
            "total_latency_seconds": round(total_latency, 3),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "details": [asdict(m) for m in self.history]
        }
