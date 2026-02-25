"""
LLM Service for Bug Report Prioritization.
Provides interface to Google Gemini API for text generation.
"""

from typing import Tuple
from google import genai
from google.genai import types
from openai import OpenAI

from config import GOOGLE_API_KEY, OPENAI_API_KEY

google_client = genai.Client(api_key=GOOGLE_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def ask_gpt(prompt: str, model: str, max_token: int = 65536) -> Tuple[str, int, int]:
    """
    Send prompt to LLM API (Google or OpenAI) and get response.
    
    Returns: (content, prompt_tokens, candidate_tokens)
    """
    if "gemini" in model:
        return ask_gemini(prompt, model, max_token)
    else:
        return ask_openai(prompt, model, max_token)


def ask_gemini(prompt: str, model: str, max_token: int) -> Tuple[str, int, int]:
    try:
        response = google_client.models.generate_content(
            model=model,
            contents={'text': prompt},
            config=types.GenerateContentConfig(
                max_output_tokens=max_token,
                temperature=0.0,
            ),
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


def ask_openai(prompt: str, model: str, max_token: int) -> Tuple[str, int, int]:
    if not openai_client:
        print("Error: OPENAI_API_KEY not set.")
        return "", 0, 0
        
    # Default to gpt-5.2 if an invalid model name might be passed or just rely on caller
    # Use user-specified model.
    
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_token if max_token < 16000 else 4096, # Adjust max tokens for OpenAI if needed, or keep high
            temperature=0.0,
        )
        
        content = response.choices[0].message.content
        
        prompt_tokens = response.usage.prompt_tokens
        candidate_tokens = response.usage.completion_tokens
        
        return content, prompt_tokens, candidate_tokens

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return "", 0, 0
