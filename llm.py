import os
from typing import Tuple

from config import create_openai_client

OPENAI_CLIENT = create_openai_client()


def ask_gpt(prompt: str, max_token: int = 4096, model: str = "gpt-5.5") -> Tuple[str, int, int]:
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    # gpt-5.x models use max_completion_tokens; older models use max_tokens
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max_token
    else:
        kwargs["max_tokens"] = max_token
    response = OPENAI_CLIENT.chat.completions.create(**kwargs)
    response_cost = response.usage.completion_tokens
    prompt_cost = response.usage.prompt_tokens
    return response.choices[0].message.content, prompt_cost, response_cost
