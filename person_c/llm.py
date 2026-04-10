# ============================================================
# FILE: person_c/llm.py  (Gemini version)
# ============================================================
# Add to requirements.txt:
#   google-generativeai
#
# Add to .env:
#   GEMINI_API_KEY=your_key_here
#   LLM_PROVIDER=gemini
#   GEMINI_MODEL=gemini-1.5-flash   # or gemini-1.5-pro / gemini-2.0-flash
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
    Unified LLM call. Routes to Gemini, OpenAI, or Anthropic
    based on LLM_PROVIDER in .env.
    """
    if LLM_PROVIDER == "gemini":
        return _call_gemini(system_prompt, user_message,
                            model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                            temperature, max_tokens)
    elif LLM_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, user_message,
                               model or "claude-3-haiku-20240307",
                               temperature, max_tokens)
    else:  # default: openai
        return _call_openai(system_prompt, user_message,
                            model or "gpt-4o-mini",
                            temperature, max_tokens)


# ── Gemini ───────────────────────────────────────────────────

def _call_gemini(system_prompt, user_message, model, temperature, max_tokens) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    client = genai.GenerativeModel(
        model_name=model,
        system_instruction=system_prompt,          # Gemini's system prompt field
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )

    response = client.generate_content(user_message)
    return response.text.strip()


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