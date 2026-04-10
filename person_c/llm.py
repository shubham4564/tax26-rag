# ============================================================
# FILE: person_c/llm.py  (multi-provider: Perplexity / Gemini / OpenAI / Anthropic)
# ============================================================
# requirements.txt needs only:
#   openai          ← used for both OpenAI AND Perplexity (same SDK)
#   google-genai    ← only if using Gemini
#   anthropic       ← only if using Anthropic
#
# .env for Perplexity:
#   LLM_PROVIDER=perplexity
#   PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxx
#   PERPLEXITY_MODEL=sonar              # see model table below
#
# Perplexity free-tier models (as of April 2026):
#   sonar                — fast, lightweight (recommended for eval runs)
#   sonar-pro            — stronger reasoning, higher cost
#   sonar-reasoning      — chain-of-thought, slower
#   sonar-reasoning-pro  — best quality, most expensive
# ============================================================

import os
from dotenv import load_dotenv
load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()


def call_llm(
    system_prompt: str,
    user_message:  str,
    model:         str | None = None,
    temperature:   float = 0.0,
    max_tokens:    int = 1024,
) -> str:
    """
    Unified LLM call. Routes based on LLM_PROVIDER in .env.
    Supported: perplexity | gemini | openai | anthropic
    """
    if LLM_PROVIDER == "perplexity":
        return _call_perplexity(system_prompt, user_message,
                                model or os.getenv("PERPLEXITY_MODEL", "sonar"),
                                temperature, max_tokens)
    elif LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_message,
                            model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                            temperature, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message,
                               model or "claude-3-haiku-20240307",
                               temperature, max_tokens)
    else:  # openai
        return _call_openai(system_prompt, user_message,
                            model or "gpt-4o-mini",
                            temperature, max_tokens)


# ── Perplexity ───────────────────────────────────────────────
# Uses the OpenAI SDK — Perplexity's API is fully compatible.

def _call_perplexity(system_prompt, user_message, model, temperature, max_tokens) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai",   # only difference from OpenAI
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Gemini ───────────────────────────────────────────────────

def _call_gemini(system_prompt, user_message, model, temperature, max_tokens) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    model_id = model.replace("models/", "")

    response = client.models.generate_content(
        model=model_id,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    # response.text is None when the model returns no candidates
    # (e.g. safety filter triggered or empty generation)
    text = response.text
    if text is None:
        print(f"  [llm WARN] Gemini returned None response for model={model_id}")
        return ""
    return text.strip()


# ── OpenAI (unchanged, kept for reference) ───────────────────

def _call_openai(system_prompt, user_message, model, temperature, max_tokens) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Anthropic (unchanged, kept for reference) ────────────────

def _call_anthropic(system_prompt, user_message, model, temperature, max_tokens) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    return resp.content[0].text.strip()


# ── Smoke test ───────────────────────────────────────────────

if __name__ == "__main__":
    reply = call_llm(
        system_prompt="You are a helpful assistant.",
        user_message="Say 'Gemini is working.' and nothing else.",
    )
    print(reply)