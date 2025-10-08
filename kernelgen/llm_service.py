#!/usr/bin/env python3
"""
Clean LLM Service supporting OpenAI, Claude, and vLLM.
"""

import os
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import httpx


@dataclass
class LLMResponse:
    """Unified response format for all LLM providers."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None

    def __str__(self):
        preview = self.content[:100] + "..." if len(self.content) > 100 else self.content
        return f"LLMResponse(provider={self.provider}, model={self.model}, content='{preview}')"


class LLMConfig:
    """Load and manage LLM configuration."""

    def __init__(self, config_path: str = "llm_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Replace environment variables
        config = self._resolve_env_vars(config)
        return config

    def _resolve_env_vars(self, obj: Any) -> Any:
        """Recursively resolve ${ENV_VAR} in config."""
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(v) for v in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj

    def get_provider_config(self, provider: Optional[str] = None) -> dict:
        """Get configuration for specified provider or default."""
        provider = provider or self.config.get("provider", "claude")

        if provider not in self.config:
            raise ValueError(f"Provider '{provider}' not configured")

        return {
            "provider": provider,
            **self.config[provider],
            "retry": self.config.get("retry", {})
        }


class BaseProvider:
    """Base class for LLM providers."""

    def __init__(self, config: dict):
        self.config = config
        self.timeout = config.get("timeout", 120)
        self.max_retries = config.get("retry", {}).get("max_retries", 3)
        self.retry_delay = config.get("retry", {}).get("retry_delay", 2)
        self.exponential_backoff = config.get("retry", {}).get("exponential_backoff", True)

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send chat completion request with retry logic."""
        last_error = None
        retry_delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return self._chat_impl(messages, **kwargs)
            except httpx.ReadTimeout as e:
                last_error = f"Request timed out after {self.timeout}s: {e}"
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    raise RuntimeError(last_error) from e
            except Exception as e:
                last_error = f"Request failed: {e}"

            # Retry logic
            if attempt < self.max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                print(f"  Error: {last_error}")  # Debug: show the actual error
                time.sleep(retry_delay)
                if self.exponential_backoff:
                    retry_delay *= 2

        raise RuntimeError(f"All {self.max_retries} attempts failed. Last error: {last_error}")

    def _chat_impl(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Implementation to be overridden by subclasses."""
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    """OpenAI API provider (GPT-5, GPT-4o, GPT-4, etc.)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-5")

        if not self.api_key:
            raise ValueError("OpenAI API key not configured")

    def _chat_impl(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send request to OpenAI API."""
        model = kwargs.get("model", self.model)

        # GPT-5 uses different endpoint and parameters
        if model == "gpt-5":
            return self._chat_gpt5(messages, **kwargs)
        else:
            return self._chat_standard(messages, **kwargs)

    def _chat_gpt5(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send request to GPT-5 API using responses.create."""
        url = f"{self.base_url}/responses"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert messages format for GPT-5
        # GPT-5 expects 'input' instead of 'messages'
        input_messages = []
        for msg in messages:
            input_messages.append({
                "role": msg.get("role"),
                "content": msg.get("content")
            })

        data = {
            "model": "gpt-5",
            "input": input_messages,
            "max_output_tokens": kwargs.get("max_output_tokens", self.config.get("max_tokens", 4096)),
            "reasoning": {
                "effort": kwargs.get("effort", self.config.get("effort", "medium"))
            }
        }

        timeout = httpx.Timeout(self.timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        # Extract content from GPT-5 response format
        content = ""
        output_items = result.get("output", [])

        for item in output_items:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue

            for content_item in item.get("content", []):
                if (
                    isinstance(content_item, dict)
                    and content_item.get("type") == "output_text"
                ):
                    content += content_item.get("text", "")

        if not content:
            # Some beta responses may expose a flattened text field
            content = result.get("output_text", "") or result.get("response", "")

        if not content:
            raise RuntimeError("GPT-5 response did not include any output text")

        return LLMResponse(
            content=content,
            model="gpt-5",
            provider="openai",
            usage=result.get("usage")  # GPT-5 returns usage under 'usage'
        )

    def _chat_standard(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send request to standard OpenAI API (GPT-4, GPT-4o, etc.)."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 4096)),
        }

        timeout = httpx.Timeout(self.timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            model=result["model"],
            provider="openai",
            usage=result.get("usage")
        )


class ClaudeProvider(BaseProvider):
    """Anthropic Claude API provider (Opus 4, Sonnet 4.5)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.anthropic.com")
        self.model = config.get("model", "claude-sonnet-4-5-20250929")

        if not self.api_key:
            raise ValueError("Claude API key not configured")

    def _chat_impl(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send request to Claude API."""
        url = f"{self.base_url}/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        # Separate system message from user/assistant messages
        system_content = ""
        user_messages = []

        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                user_messages.append(msg)

        # Check if thinking is enabled first to determine temperature
        thinking = kwargs.get("thinking", self.config.get("thinking", "enabled"))

        # Temperature must be 1 when thinking is enabled
        if thinking == "enabled":
            temperature = 1
        else:
            temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))

        data = {
            "model": kwargs.get("model", self.model),
            "messages": user_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 4096)),
            "temperature": temperature,
        }

        # Add thinking parameter if configured
        if thinking in ["enabled", "disabled"]:
            if thinking == "enabled":
                # When enabled, budget_tokens is required (minimum 1024)
                # Default to 10000 tokens for thinking budget
                budget_tokens = kwargs.get("thinking_budget", self.config.get("thinking_budget", 10000))
                data["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
            else:
                data["thinking"] = {"type": "disabled"}

        if system_content:
            data["system"] = system_content

        timeout = httpx.Timeout(self.timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        # Extract content from Claude response
        content = ""
        if "content" in result and isinstance(result["content"], list):
            if result["content"] and "text" in result["content"][0]:
                content = result["content"][0]["text"]

        return LLMResponse(
            content=content,
            model=result.get("model", self.model),
            provider="claude",
            usage=result.get("usage")
        )


class VLLMProvider(BaseProvider):
    """vLLM local model provider (OpenAI-compatible API)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:8000/v1")
        self.model = config.get("model", "local-model")
        self.api_key = config.get("api_key", None)  # Optional API key for vLLM

    def _chat_impl(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """Send request to vLLM server (OpenAI-compatible)."""
        url = f"{self.base_url}/chat/completions"

        headers = {
            "Content-Type": "application/json"
        }

        # Add API key if configured
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.get("temperature", 0.7)),
            "max_tokens": kwargs.get("max_tokens", self.config.get("max_tokens", 4096)),
        }

        timeout = httpx.Timeout(self.timeout)

        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()

        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            model=result.get("model", self.model),
            provider="vllm",
            usage=result.get("usage")
        )


class LLMService:
    """Main LLM service interface."""

    PROVIDERS = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "vllm": VLLMProvider,
    }

    def __init__(self, config_path: str = "llm_config.yaml", provider: Optional[str] = None):
        """
        Initialize LLM service.

        Args:
            config_path: Path to YAML config file
            provider: Provider to use (openai/claude/vllm), or None for default from config
        """
        self.config_manager = LLMConfig(config_path)
        self.provider_name = provider
        self._provider = None

    @property
    def provider(self) -> BaseProvider:
        """Lazy load provider."""
        if self._provider is None:
            config = self.config_manager.get_provider_config(self.provider_name)
            provider_class = self.PROVIDERS.get(config["provider"])

            if not provider_class:
                raise ValueError(f"Unknown provider: {config['provider']}")

            self._provider = provider_class(config)

        return self._provider

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        """
        Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Override config parameters (temperature, max_tokens, model, etc.)

        Returns:
            LLMResponse with content, model, provider info
        """
        return self.provider.chat(messages, **kwargs)

    def simple_chat(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """
        Simple single-turn chat.

        Args:
            prompt: User prompt
            system: Optional system message
            **kwargs: Additional parameters

        Returns:
            Response content as string
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages, **kwargs)
        return response.content


# Example usage
if __name__ == "__main__":
    # Initialize service (uses default provider from config)
    service = LLMService("llm_config.yaml")

    # Simple chat
    response = service.simple_chat(
        prompt="What is 2+2?",
        system="You are a helpful math assistant."
    )
    print(f"Response: {response}")

    # Multi-turn conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    result = service.chat(messages)
    print(f"\n{result}")

    # Use openai provider
    # openai_service = LLMService("llm_config.yaml", provider="openai")
    # response = openai_service.simple_chat("Hello!")
    # print(f"\nOpenAI: {response}")
