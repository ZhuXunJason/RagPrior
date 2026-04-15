import os
from typing import Tuple

from config import create_openai_client

OPENAI_CLIENT = create_openai_client()


def ask_gpt(prompt: str, max_token: int = 4096) -> Tuple[str, int, int]:
    response = OPENAI_CLIENT.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_token,
    )
    response_cost = response.usage.completion_tokens
    prompt_cost = response.usage.prompt_tokens
    return response.choices[0].message.content, prompt_cost, response_cost
