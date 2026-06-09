"""
Module: context_manager.py
Description: The state management layer. Aggregates data from active Jira inputs, 
             maintains a localized JSON history log to simulate long-term memory, 
             and provides a unified execution interface.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from prompt_orchestrator import JiraTicket, ProductHistory, UserContext, DynamicPromptOrchestrator
from llm_connector import LLMConnector, SegmentedResponse


# ==========================================
# 1. LOCAL DATA PERSISTENCE LAYER
# ==========================================

class LocalContextStore:
    def __init__(self, storage_path: str = "data_store.json"):
        self.storage_path = storage_path
        self._initialize_store()

    def _initialize_store(self):
        """Creates a default local JSON data store file if it does not exist."""
        if not os.path.exists(self.storage_path):
            default_structure = {
                "historical_constraints": [
                    "Must ensure backward compatibility with Python 3.10+",
                    "All microservices must communicate via async gRPC channels"
                ],
                "previously_resolved_issues": [],
                "past_generated_features": [],
                "execution_logs": []
            }
            with open(self.storage_path, 'w') as f:
                json.dump(default_structure, f, indent=4)

    def load_state(self) -> Dict[str, Any]:
        """Reads the current localized storage state from disk."""
        with open(self.storage_path, 'r') as f:
            return json.load(f)

    def append_log(self, ticket_id: str, prompt: str, response: SegmentedResponse):
        """Logs an execution transaction and automatically updates product history."""
        state = self.load_state()
        
        # Log transaction metrics
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "ticket_id": ticket_id,
            "provider": response.metrics.provider_used,
            "model": response.metrics.model_name,
            "latency_seconds": response.metrics.execution_time_seconds
        }
        state["execution_logs"].append(log_entry)
        
        # Append successful completions directly back into our feature memory track
        state["past_generated_features"].append(f"Implemented solution for ticket {ticket_id}")
        
        with open(self.storage_path, 'w') as f:
            json.dump(state, f, indent=4)


# ==========================================
# 2. UNIFIED PIPELINE ORCHESTRATOR
# ==========================================

class ContextPipelineCoordinator:
    def __init__(self, provider: str = "mock", storage_path: str = "data_store.json"):
        self.store = LocalContextStore(storage_path)
        self.orchestrator = DynamicPromptOrchestrator()
        self.connector = LLMConnector(provider=provider)

    def run_ticket_pipeline(self, raw_ticket_data: Dict[str, Any], active_user: UserContext) -> SegmentedResponse:
        """
        Executes the entire end-to-end processing loop:
        1. Compiles raw dictionary into validated Pydantic models.
        2. Injects system constraints pulled from local data storage.
        3. Constructs the dynamic prompt template block.
        4. Triggers LLM inference execution.
        5. Logs the results back to disk for ongoing context tracking.
        """
        # Load local storage parameters
        current_state = self.store.load_state()
        
        # Step 1: Instantiate core validated models
        ticket = JiraTicket(
            ticket_id=raw_ticket_data.get("id", "UNKNOWN"),
            summary=raw_ticket_data.get("summary", ""),
            description=raw_ticket_data.get("description", ""),
            acceptance_criteria=raw_ticket_data.get("criteria", [])
        )
        
        history = ProductHistory(
            architectural_constraints=current_state.get("historical_constraints", []),
            previous_features=current_state.get("past_generated_features", []),
            resolved_issues=current_state.get("previously_resolved_issues", [])
        )
        
        # Step 2: Compile the structured template payload
        compiled_prompt = self.orchestrator.generate_prompt(
            ticket=ticket, 
            history=history, 
            context=active_user
        )
        
        # Step 3: Run pipeline inference matching selected provider engine
        response = self.connector.execute(prompt=compiled_prompt)
        
        # Step 4: Write state mutations back to local disk
        self.store.append_log(ticket_id=ticket.ticket_id, prompt=compiled_prompt, response=response)
        
        return response


# ==========================================
# 3. RUNTIME PIPELINE VERIFICATION
# ==========================================

if __name__ == "__main__":
    print("--- [INITIALIZING CONTEXT COORDINATOR STREAM] ---")
    
    # Simulate an incoming webhook payload from a software Jira board
    incoming_jira_webhook = {
        "id": "ENG-707",
        "summary": "Implement Redis Session Cache Cluster",
        "description": "Configure an multi-node replication setup to manage shared worker states securely.",
        "criteria": [
            "Must replicate session states across 3 visual sub-nodes",
            "Encryption in transit via TLS required"
        ]
    }
    
    # Establish active user context parameters
    current_session_user = UserContext(
        user_role="Senior DevOps Architect",
        preferences={"security_strictness": "Maximum"}
    )
    
    # Run pipeline coordinator
    coordinator = ContextPipelineCoordinator(provider="mock")
    pipeline_result = coordinator.run_ticket_pipeline(
        raw_ticket_data=incoming_jira_webhook, 
        active_user=current_session_user
    )
    
    print(f"Status: Executed | Logs Committed to Storage Path.")
    print(f"Time Taken: {pipeline_result.metrics.execution_time_seconds}s")
    print(f"\n[OUTPUT FROM LOCALIZED MEMORY PIPELINE]:\n{pipeline_result.final_output}")