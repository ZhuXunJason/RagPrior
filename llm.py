from typing import Any, Dict, Optional, Tuple

from google import genai
from google.genai import types
from openai import OpenAI

from config import create_openai_client, get_base_url, get_google_api_key, get_openai_api_key


def is_gemini_model(model: str) -> bool:
    return "gemini" in model.lower()


def _build_gemini_client(api_key: Optional[str] = None) -> Optional[genai.Client]:
    effective_api_key = api_key or get_google_api_key()
    if not effective_api_key:
        return None
    return genai.Client(api_key=effective_api_key)


def _build_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAI:
    if api_key is None and base_url is None:
        return create_openai_client()

    kwargs: Dict[str, Any] = {}
    effective_api_key = api_key or get_openai_api_key()
    effective_base_url = base_url or get_base_url()

    if effective_api_key:
        kwargs["api_key"] = effective_api_key
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    return OpenAI(**kwargs)


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "".join(parts)
    return str(content)


def ask_completion(
    prompt: str,
    model: str,
    max_token: int = 4096,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if is_gemini_model(model):
        return ask_gemini(
            prompt,
            model,
            max_token,
            api_key=api_key,
            request_options=request_options,
            temperature=temperature,
        )
    return ask_openai(
        prompt,
        model,
        max_token,
        base_url=base_url,
        api_key=api_key,
        request_options=request_options,
        temperature=temperature,
    )


def ask_gpt(
    prompt: str,
    max_token: int = 4096,
    model: str = "gpt-5.5",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    result = ask_completion(
        prompt,
        model=model,
        max_token=max_token,
        temperature=0.0,
        base_url=base_url,
        api_key=api_key,
        request_options=request_options,
    )
    return (
        result.get("content", ""),
        result.get("prompt_tokens", 0),
        result.get("completion_tokens", 0),
    )


def ask_gemini(
    prompt: str,
    model: str,
    max_token: int,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    gemini_client = _build_gemini_client(api_key=api_key)
    if not gemini_client:
        raise RuntimeError("GOOGLE_API_KEY not set.")

    config_kwargs: Dict[str, Any] = {
        "max_output_tokens": max_token,
        "temperature": temperature,
    }
    if request_options:
        config_kwargs.update(request_options.get("gemini_config", {}))

    response = gemini_client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    content = ""
    try:
        content = response.text or ""
    except ValueError:
        content = ""

    candidate = None
    if getattr(response, "candidates", None):
        candidate = response.candidates[0]

    if not content and candidate is not None:
        content = _normalize_content(getattr(candidate.content, "parts", None))

    usage = getattr(response, "usage_metadata", None)
    finish_reason = getattr(candidate, "finish_reason", None)

    return {
        "content": content,
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
        "completion_tokens": getattr(usage, "candidates_token_count", 0) or 0,
        "finish_reason": str(finish_reason) if finish_reason is not None else "",
        "provider": "gemini",
    }


def ask_openai(
    prompt: str,
    model: str,
    max_token: int,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    openai_client = _build_openai_client(api_key=api_key, base_url=base_url)

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if model.startswith("gpt-5"):
        request_kwargs["max_completion_tokens"] = max_token
    else:
        request_kwargs["max_tokens"] = max_token
    if request_options:
        if request_options.get("extra_body"):
            request_kwargs["extra_body"] = request_options["extra_body"]
        if request_options.get("stream_options"):
            request_kwargs["stream_options"] = request_options["stream_options"]

    response = openai_client.chat.completions.create(**request_kwargs)
    choice = response.choices[0] if response.choices else None
    content = _normalize_content(choice.message.content if choice else "")

    return {
        "content": content,
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
        "finish_reason": getattr(choice, "finish_reason", "") or "",
        "provider": "openai",
    }
