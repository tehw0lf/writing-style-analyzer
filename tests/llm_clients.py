"""
LLM Client Abstraction for Testing

Provides a unified interface for testing with multiple LLM providers:
- Anthropic (Claude)
- OpenAI (GPT models)
- Open WebUI (Ollama/local models via OpenAI-compatible API)

All clients are optional - they gracefully skip if API keys are not configured.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMClient(ABC):
    """Base class for LLM clients"""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, timeout: int = 60) -> str:
        """Generate text from prompt"""
        pass

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check if API key is configured"""
        pass


class AnthropicClient(LLMClient):
    """Anthropic/Claude API client"""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        super().__init__(model, max_tokens, temperature)
        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: uv pip install anthropic"
            )

    def generate(self, prompt: str, timeout: int = 60) -> str:
        """Generate text using Claude"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        return message.content[0].text

    @classmethod
    def is_available(cls) -> bool:
        """Check if Anthropic API key is configured"""
        if os.getenv("ANTHROPIC_API_KEY") is None:
            return False
        try:
            import anthropic

            return True
        except ImportError:
            return False


class OpenAIClient(LLMClient):
    """OpenAI API client"""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        super().__init__(model, max_tokens, temperature)
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: uv pip install openai"
            )

    def generate(self, prompt: str, timeout: int = 60) -> str:
        """Generate text using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        return response.choices[0].message.content

    @classmethod
    def is_available(cls) -> bool:
        """Check if OpenAI API key is configured"""
        if os.getenv("OPENAI_API_KEY") is None:
            return False
        try:
            from openai import OpenAI

            return True
        except ImportError:
            return False


class OpenWebUIClient(LLMClient):
    """Open WebUI / Ollama client (OpenAI-compatible API)"""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        super().__init__(model, max_tokens, temperature)
        self.base_url = os.getenv("OPENWEBUI_BASE_URL")
        if not self.base_url:
            raise ValueError(
                "OPENWEBUI_BASE_URL environment variable required for Open WebUI"
            )

        try:
            from openai import OpenAI

            # Open WebUI uses OpenAI-compatible API format
            # API key can be dummy for local Ollama instances
            api_key = os.getenv("OPENWEBUI_API_KEY", "dummy-key-for-local")
            self.client = OpenAI(base_url=self.base_url, api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: uv pip install openai"
            )

    def generate(self, prompt: str, timeout: int = 60) -> str:
        """Generate text using Open WebUI / Ollama"""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout,
        )
        return response.choices[0].message.content

    @classmethod
    def is_available(cls) -> bool:
        """Check if Open WebUI is configured"""
        base_url = os.getenv("OPENWEBUI_BASE_URL")
        if base_url is None:
            return False
        try:
            from openai import OpenAI

            return True
        except ImportError:
            return False


def get_available_clients(config: dict) -> dict[str, LLMClient]:
    """
    Get all available LLM clients based on configuration and API keys.

    Args:
        config: LLM validation config from tests/config.yaml

    Returns:
        Dictionary of provider_name -> LLMClient instance for available providers
    """
    providers = config["providers"]
    available = {}

    # Check Anthropic
    if providers["anthropic"]["enabled"] and AnthropicClient.is_available():
        available["anthropic"] = AnthropicClient(
            model=providers["anthropic"]["model"],
            max_tokens=providers["anthropic"]["max_tokens"],
            temperature=providers["anthropic"]["temperature"],
        )

    # Check OpenAI
    if providers["openai"]["enabled"] and OpenAIClient.is_available():
        available["openai"] = OpenAIClient(
            model=providers["openai"]["model"],
            max_tokens=providers["openai"]["max_tokens"],
            temperature=providers["openai"]["temperature"],
        )

    # Check Open WebUI
    if providers["openwebui"]["enabled"] and OpenWebUIClient.is_available():
        available["openwebui"] = OpenWebUIClient(
            model=providers["openwebui"]["model"],
            max_tokens=providers["openwebui"]["max_tokens"],
            temperature=providers["openwebui"]["temperature"],
        )

    return available
