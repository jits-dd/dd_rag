from typing import List, Dict, Any
import json
import ast
from config.settings import settings
import logging

class AgentOrchestrator:
    def __init__(self, agents: list):
        self.agents = agents
        self.logger = logging.getLogger(__name__)

    def _parse_json(self, input_string):
        """Robust JSON parsing"""
        try:
            cleaned = input_string.strip()
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:-3].strip()
            return json.loads(cleaned)
        except Exception as e:
            self.logger.error(f"JSON parsing failed: {e}")
            raise ValueError("Invalid agent response format")

    def _select_agent(self, query: str) -> Dict[str, str]:
        """Agent selection with constrained output"""
        prompt = f"""
        Analyze the query and select exactly ONE agent.
        
        Available Agents:
        {", ".join([f"{agent.name} ({agent.description})" for agent in self.agents])}
        
        Query: {query}
        
        Respond ONLY with valid JSON in this format:
        {{
            "agent": "agent_name",
            "reason": "explanation",
            "rewritten_query": "optimized_query"
        }}
        """

        try:
            response = settings.llm.complete(prompt)
            return self._parse_json(str(response))
        except Exception as e:
            self.logger.error(f"Agent selection failed: {e}")
            return {
                "agent": self.agents[0].name,
                "reason": "Fallback due to selection error",
                "rewritten_query": query
            }

    def orchestrate_task(self, query: str) -> Dict[str, Any]:
        """Complete orchestration with proper error handling"""
        try:
            print(f"\nOrchestrator received query: {query}")

            # Step 1: Select agent
            agent_selection = self._select_agent(query)
            print(f"Selected agent: {agent_selection}")

            # Step 2: Execute with selected agent
            for agent in self.agents:
                if agent.name == agent_selection["agent"]:
                    print(f"Executing with agent: {agent.name}")
                    result = agent.process(agent_selection["rewritten_query"])

                    if not result:
                        raise ValueError(f"Agent {agent.name} returned empty result")

                    result["agent"] = agent.name
                    result["reason"] = agent_selection["reason"]
                    return result

            # Fallback if no agent matched
            return {
                "answer": "No suitable agent found to handle this query",
                "agents_available": [agent.name for agent in self.agents]
            }

        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            return {
                "answer": "System error during query processing",
                "error": str(e)
            }