"""
Module: llm_connector.py
Description: Multi-backend LLM execution layer. Supports Groq (free, fast),
             Ollama (local), Anthropic (cloud), and mock providers.
"""

import time
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ==========================================
# 1. DATA SCHEMAS (Response Structures)
# ==========================================

class ProviderType(str, Enum):
    """Supported LLM backends."""
    GROQ = "groq"
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    MOCK = "mock"


@dataclass
class ExecutionMetrics:
    """Encapsulates performance and backend metadata."""
    provider_used: str
    model_name: str
    execution_time_seconds: float
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None


@dataclass
class SegmentedResponse:
    """Structured response from LLM connector."""
    final_output: str
    metrics: ExecutionMetrics
    reasoning_chain: Optional[str] = None
    structured_data: Dict[str, Any] = field(default_factory=dict)


# ==========================================
# 2. BACKEND IMPLEMENTATIONS
# ==========================================

class GroqBackend:
    """Groq backend - free, fast Llama 3.3 70B inference."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = "llama-3.3-70b-versatile"
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable or "
                "pass api_key to GroqBackend(). Get free key at console.groq.com"
            )
        
        # Lazy import to avoid hard dependency
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        except ImportError:
            raise ImportError("Groq SDK not installed. Run: pip install groq")

    def execute(self, prompt: str) -> SegmentedResponse:
        """Execute prompt via Groq API."""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            
            output = response.choices[0].message.content
            latency = time.time() - start_time
            
            return SegmentedResponse(
                final_output=output,
                metrics=ExecutionMetrics(
                    provider_used="groq",
                    model_name=self.model,
                    execution_time_seconds=latency,
                    token_count=response.usage.total_tokens if hasattr(response, 'usage') else None,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Groq execution failed: {e}")


class OllamaBackend:
    """Ollama backend - local inference via HTTP."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        
        try:
            import httpx
            self.client = httpx.Client(base_url=self.base_url, timeout=120.0)
        except ImportError:
            raise ImportError("httpx not installed. Run: pip install httpx")

    def execute(self, prompt: str) -> SegmentedResponse:
        """Execute prompt via Ollama local API."""
        start_time = time.time()
        
        try:
            response = self.client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                }
            )
            response.raise_for_status()
            data = response.json()
            
            output = data.get("response", "")
            latency = time.time() - start_time
            
            return SegmentedResponse(
                final_output=output,
                metrics=ExecutionMetrics(
                    provider_used="ollama",
                    model_name=self.model,
                    execution_time_seconds=latency,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Ollama execution failed: {e}")


class AnthropicBackend:
    """Anthropic backend - Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or "
                "pass api_key to AnthropicBackend()."
            )
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")

    def execute(self, prompt: str, system: str = "") -> SegmentedResponse:
        """Execute prompt via Anthropic API."""
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system or "You are a helpful AI assistant.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            output = response.content[0].text
            latency = time.time() - start_time
            
            return SegmentedResponse(
                final_output=output,
                metrics=ExecutionMetrics(
                    provider_used="anthropic",
                    model_name=self.model,
                    execution_time_seconds=latency,
                    token_count=response.usage.output_tokens if hasattr(response, 'usage') else None,
                )
            )
        except Exception as e:
            raise RuntimeError(f"Anthropic execution failed: {e}")


class MockBackend:
    """Mock backend for testing - returns deterministic responses."""
    
    def execute(self, prompt: str) -> SegmentedResponse:
        """Execute prompt via mock (instant, deterministic)."""
        start_time = time.time()
        latency = time.time() - start_time
        
        output = f"""### MOCK RESPONSE
Analyzed Ticket: [Parsed from prompt]
Recommended Approach:
1. Design Phase: Establish architecture aligned with product constraints
2. Implementation: Execute in staged deployment (dev → staging → prod)
3. Validation: Run acceptance criteria verification before rollout
4. Documentation: Update runbooks and architectural diagrams
Estimated Effort: 8-12 engineering hours
Risk Assessment: LOW (aligns with historical constraints)
"""
        
        return SegmentedResponse(
            final_output=output,
            metrics=ExecutionMetrics(
                provider_used="mock",
                model_name="mock-v1",
                execution_time_seconds=latency,
            )
        )


# ==========================================
# 3. UNIFIED LLM CONNECTOR
# ==========================================

class LLMConnector:
    """Unified interface for multi-backend LLM execution."""
    
    BACKENDS = {
        ProviderType.GROQ: GroqBackend,
        ProviderType.OLLAMA: OllamaBackend,
        ProviderType.ANTHROPIC: AnthropicBackend,
        ProviderType.MOCK: MockBackend,
    }

    def __init__(self, provider: str = "mock", **kwargs):
        """
        Initialize LLMConnector with chosen backend.
        
        Args:
            provider: One of "groq", "ollama", "anthropic", "mock"
            **kwargs: Backend-specific arguments (api_key, model, base_url, etc.)
        """
        self.provider = ProviderType(provider.lower())
        
        backend_class = self.BACKENDS.get(self.provider)
        if not backend_class:
            raise ValueError(f"Unknown provider: {provider}")
        
        self.backend = backend_class(**kwargs)

    def execute(self, prompt: str, system_instructions: str = "") -> SegmentedResponse:
        """
        Execute prompt against the configured LLM backend.
        
        Args:
            prompt: The user/execution prompt to send to the LLM
            system_instructions: Optional system context (used by Anthropic)
            
        Returns:
            SegmentedResponse with output, metrics, and optional structured data
        """
        # Special handling for Anthropic which expects system separately
        if self.provider == ProviderType.ANTHROPIC:
            return self.backend.execute(prompt, system=system_instructions)
        
        # Other backends integrate system into prompt or ignore it
        return self.backend.execute(prompt)


# ==========================================
# 4. RUNTIME VERIFICATION (Smoke Test)
# ==========================================

if __name__ == "__main__":
    print("--- [LLM CONNECTOR SMOKE TEST] ---\n")
    
    # Test 1: Mock backend (always works)
    print("[1/3] Testing Mock Backend...")
    try:
        mock_connector = LLMConnector(provider="mock")
        mock_result = mock_connector.execute("Sample prompt")
        print(f"✅ Mock backend: {mock_result.metrics.provider_used} | "
              f"Latency: {mock_result.metrics.execution_time_seconds:.3f}s")
        print(f"Output preview: {mock_result.final_output[:150]}...\n")
    except Exception as e:
        print(f"❌ Mock backend failed: {e}\n")

    # Test 2: Ollama backend (requires local Ollama)
    print("[2/3] Testing Ollama Backend...")
    try:
        ollama_connector = LLMConnector(provider="ollama")
        ollama_result = ollama_connector.execute("What is a Jira ticket?")
        print(f"✅ Ollama backend: {ollama_result.metrics.provider_used} | "
              f"Model: {ollama_result.metrics.model_name} | "
              f"Latency: {ollama_result.metrics.execution_time_seconds:.3f}s")
    except Exception as e:
        print(f"⚠️  Ollama backend unavailable (expected if Ollama not running): {e}\n")

    # Test 3: Groq backend (requires GROQ_API_KEY env var)
    print("[3/3] Testing Groq Backend...")
    try:
        groq_connector = LLMConnector(provider="groq")
        groq_result = groq_connector.execute("What is a Jira ticket?")
        print(f"✅ Groq backend: {groq_result.metrics.provider_used} | "
              f"Model: {groq_result.metrics.model_name} | "
              f"Latency: {groq_result.metrics.execution_time_seconds:.3f}s")
    except Exception as e:
        print(f"⚠️  Groq backend unavailable (expected if GROQ_API_KEY not set): {e}\n")

    print("\n--- [CONNECTOR STATUS: READY] ---")
    print("To use Groq: export GROQ_API_KEY=gsk_... (get free key at console.groq.com)")
    print("To use Ollama: ollama serve (in another terminal)")
    print("To use Anthropic: export ANTHROPIC_API_KEY=sk-ant-...")
