"""
Module: prompt_orchestrator.py
Description: Enterprise-level dynamic prompt construction pipeline merging 
             Jira tickets, product history, and user context.
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field


# ==========================================
# 1. DATA SCHEMAS (Input Validation)
# ==========================================

class JiraTicket(BaseModel):
    ticket_id: str = Field(..., description="Unique identifier (e.g., JIRA-101)")
    summary: str
    description: str
    acceptance_criteria: List[str] = Field(default_factory=list)


class ProductHistory(BaseModel):
    previous_features: List[str] = Field(default_factory=list)
    resolved_issues: List[str] = Field(default_factory=list)
    architectural_constraints: List[str] = Field(default_factory=list)


class UserContext(BaseModel):
    user_role: str
    interaction_history: List[str] = Field(default_factory=list)
    preferences: Dict[str, str] = Field(default_factory=dict)


# ==========================================
# 2. PROMPT TEMPLATE BLUEPRINTS
# ==========================================

SYSTEM_INSTRUCTIONS = """You are an advanced, context-aware engineering assistant. 
Your task is to analyze target Jira tickets and generate optimal outputs by aligning them against the product history and specific user constraints.
Follow the Chain-of-Thought reasoning breakdown before outputting your definitive execution plan."""

FEW_SHOT_EXAMPLES = [
    {
        "input": "Example Ticket input details...",
        "thought": "Thinking process: 1. Identify core requirement. 2. Verify against constraints...",
        "output": "Example expected high-quality output structural block."
    }
]


# ==========================================
# 3. CORE ORCHESTRATION ENGINE
# ==========================================

class DynamicPromptOrchestrator:
    def __init__(self, system_template: str = SYSTEM_INSTRUCTIONS):
        self.system_template = system_template
        self.few_shot_examples: List[Dict[str, str]] = FEW_SHOT_EXAMPLES

    def _compile_few_shot_block(self) -> str:
        """Formats stored few-shot examples into a clean string block."""
        block = "## FEW-SHOT EXAMPLES\n"
        for i, example in enumerate(self.few_shot_examples, 1):
            block += f"### Example {i}\n"
            block += f"#### Input:\n{example['input']}\n"
            block += f"#### Reasoning (Chain-of-Thought):\n{example['thought']}\n"
            block += f"#### Final Output:\n{example['output']}\n\n"
        return block

    def _compile_dynamic_context(self, ticket: JiraTicket, history: ProductHistory, context: UserContext) -> str:
        """Assembles real-time data inputs into explicit system boundary blocks."""
        return f"""## DYNAMIC RUNTIME CONTEXT

### 1. TARGET JIRA TICKET [{ticket.ticket_id}]
- **Summary:** {ticket.summary}
- **Description:** {ticket.description}
- **Acceptance Criteria:** {chr(10).join([f'  - {ac}' for ac in ticket.acceptance_criteria])}

### 2. HISTORICAL PRODUCT CONTEXT
- **Existing Architecture Constraints:** {', '.join(history.architectural_constraints)}
- **Relevant Past Features:** {', '.join(history.previous_features)}

### 3. ACTIVE USER CONTEXT
- **Operator Role:** {context.user_role}
- **Stored Behavioral Preferences:** {context.preferences}
"""

    def generate_prompt(self, ticket: JiraTicket, history: ProductHistory, context: UserContext) -> str:
        """Constructs the absolute finalized execution prompt string."""
        system_base = f"# SYSTEM INSTRUCTION\n{self.system_template}\n\n"
        few_shot_base = self._compile_few_shot_block()
        context_base = self._compile_dynamic_context(ticket, history, context)
        
        execution_trigger = "\n## EXECUTION MANDATE\nAnalyze the data compiled above. Execute a detailed, step-by-step Chain-of-Thought response matching the expected structural format."
        
        return f"{system_base}{few_shot_base}{context_base}{execution_trigger}"


# ==========================================
# 4. RUNTIME VERIFICATION (Smoke Test)
# ==========================================

if __name__ == "__main__":
    # Initialize components with mock verification data
    sample_ticket = JiraTicket(
        ticket_id="ENG-404",
        summary="Implement offline-first vector local caching",
        description="Optimize retrieval speeds by utilizing local index strategies to avoid round-trip network lag.",
        acceptance_criteria=["Must run locally without remote APIs", "Latency under 15ms"]
    )
    
    sample_history = ProductHistory(
        architectural_constraints=["Local runtime parity required", "Strict low-memory boundaries"]
    )
    
    sample_context = UserContext(
        user_role="Lead AI Architect",
        preferences={"output_style": "Tactical/Technical"}
    )
    
    # Run Orchestrator
    orchestrator = DynamicPromptOrchestrator()
    final_prompt = orchestrator.generate_prompt(sample_ticket, sample_history, sample_context)
    
    print("--- [INITIALIZATION SMOKE TEST: GENERATED PROMPT PREVIEW] ---")
    print(final_prompt[:800] + "\n\n[... Remaining prompt context truncated for preview ...] ")