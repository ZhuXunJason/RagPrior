"""
LLM Service for Bug Report Prioritization.
Provides interface to Google Gemini API for text generation.
"""

from typing import Tuple
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)


def ask_gpt(prompt: str, model: str, max_token: int = 65536) -> Tuple[str, int, int]:
    """
    Send prompt to Gemini API and get response.
    
    Returns: (content, prompt_tokens, candidate_tokens)
    """
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
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
