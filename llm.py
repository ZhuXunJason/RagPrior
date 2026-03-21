"""
LLM Service for Bug Report Prioritization.
Provides interface to Google Gemini API for text generation.
"""

from typing import Any, Dict, Optional, Tuple
from google import genai
from google.genai import types
from openai import OpenAI

from config import GOOGLE_API_KEY, OPENAI_API_KEY, BASE_URL

google_client = genai.Client(api_key=GOOGLE_API_KEY)


def _build_openai_client(api_key: Optional[str] = None, base_url: Optional[str] = None) -> Optional[OpenAI]:
    effective_api_key = api_key or OPENAI_API_KEY
    if not effective_api_key:
        return None
    kwargs = {"api_key": effective_api_key}
    
    effective_base_url = base_url or BASE_URL
    if effective_base_url:
        kwargs["base_url"] = effective_base_url
    return OpenAI(**kwargs)


def ask_gpt(
    prompt: str,
    model: str,
    max_token: int = 65536,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    """
    Send prompt to LLM API (Google or OpenAI) and get response.
    
    Returns: (content, prompt_tokens, candidate_tokens)
    """
    model_lower = model.lower()

    if "gemini" in model_lower:
        return ask_gemini(prompt, model, max_token, request_options=request_options)
    if "gpt" in model_lower:
        return ask_openai(
            prompt,
            model,
            max_token,
            base_url=base_url,
            api_key=api_key,
            request_options=request_options,
        )

    # Custom model: call with OpenAI-compatible API under the given base_url.
    if not base_url and not BASE_URL:
        print("Warning: custom model detected but base_url is not set; falling back to default OpenAI endpoint.")
    return ask_openai(
        prompt,
        model,
        max_token,
        base_url=base_url,
        api_key=api_key,
        request_options=request_options,
    )


def ask_gemini(
    prompt: str,
    model: str,
    max_token: int,
    request_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    try:
        config_kwargs: Dict[str, Any] = {
            "max_output_tokens": max_token,
            "temperature": 0.0,
        }
        if request_options:
            config_kwargs.update(request_options.get("gemini_config", {}))
        response = google_client.models.generate_content(
            model=model,
            contents={'text': prompt},
            config=types.GenerateContentConfig(**config_kwargs),
        )
        
        # Extract content
        try:
            content = response.text
        except ValueError:
            print("Warning: Response blocked or empty.")
            content = ""
            if response.candidates and response.candidates[0].content.parts:
                content = str(response.candidates[0].content.parts[0].text)

        # Gemini API usage metadata
        prompt_tokens = 0
        candidate_tokens = 0
        
        usage = getattr(response, 'usage_metadata', None)
        if usage:
            prompt_tokens = usage.prompt_token_count
            candidate_tokens = usage.candidates_token_count
            
        return content, prompt_tokens, candidate_tokens

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "", 0, 0


def ask_openai(
    prompt: str,
    model: str,
    max_token: int,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    request_options: Optional[Dict[str, Any]] = None,
) -> Tuple[str, int, int]:
    openai_client = _build_openai_client(api_key=api_key, base_url=base_url)
    if not openai_client:
        print("Error: OPENAI_API_KEY not set.")
        return "", 0, 0
        
    # Default to gpt-5.2 if an invalid model name might be passed or just rely on caller
    # Use user-specified model.
    
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_token if max_token < 16000 else 4096,
        "temperature": 0.0,
    }
    if request_options:
        if request_options.get("extra_body"):
            request_kwargs["extra_body"] = request_options["extra_body"]
        if request_options.get("stream_options"):
            request_kwargs["stream_options"] = request_options["stream_options"]

    try:
        response = openai_client.chat.completions.create(**request_kwargs)
    except Exception as e:
        if request_kwargs.get("extra_body"):
            fallback_kwargs = dict(request_kwargs)
            fallback_kwargs.pop("extra_body", None)
            try:
                response = openai_client.chat.completions.create(**fallback_kwargs)
            except Exception as inner_e:
                print(f"Error calling OpenAI API: {inner_e}")
                return "", 0, 0
        else:
            print(f"Error calling OpenAI API: {e}")
            return "", 0, 0

    try:
        content = response.choices[0].message.content
        
        prompt_tokens = response.usage.prompt_tokens
        candidate_tokens = response.usage.completion_tokens
        
        return content, prompt_tokens, candidate_tokens

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "", 0, 0
