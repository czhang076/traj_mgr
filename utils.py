"""
Utility functions for transforming mini-swe-agent trajectories to SFT format.

Supports three output styles:
  - nemotron:       Nemotron-3-Super chat template (reasoning_content on all assistant turns)
  - qwen35:         Qwen 3.5 chat template, no-think mode (reasoning stripped)
  - qwen35_think:   Qwen 3.5 chat template, think mode (reasoning_content on all turns,
                    template handles rolling truncation)

Reference chat templates:
  - Nemotron: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
  - Qwen 3.5: Qwen/Qwen3.5-122B-A22B

Both templates:
  - Use ChatML framing (<|im_start|> / <|im_end|>)
  - Render tool_calls as XML: <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
  - Wrap tool responses in <tool_response>...</tool_response> inside user role
  - Require tool_calls[].function.arguments to be a dict (Jinja2 |items filter)
  - Natively read the `reasoning_content` field on assistant messages

Key differences:
  - Nemotron: every assistant turn MUST open with <think>...</think> (empty if no CoT).
    The template's `truncate_history_thinking` (default True) auto-strips historical think
    content to <think></think>, keeping full CoT only on the last assistant turn.
  - Qwen 3.5: only assistant turns AFTER `last_query_index` (last real user message, not
    tool_response) receive <think> blocks. Earlier assistant turns get no think tags at all.
    For multi-turn SFT, Qwen recommends either no-think mode or single-turn splitting.
"""

import json
from typing import Optional

# Fields on a message that are NOT part of the standard OpenAI chat format
# and must be removed before passing to apply_chat_template.
_STRIP_KEYS = frozenset(
    ["function_call", "provider_specific_fields", "extra", "index"]
)

# Standard roles accepted by both templates
_VALID_ROLES = frozenset(["system", "user", "assistant", "tool"])


def get_messages(traj: dict) -> list[dict]:
    """Extract and normalise the message list from a mini-swe-agent v2 trajectory.

    Performs:
      1. Filters out non-standard roles (e.g. ``exit``).
      2. Strips non-standard fields added by litellm / mini-swe-agent.
      3. Converts ``tool_calls[].function.arguments`` from JSON string to dict
         (required by both Nemotron and Qwen Jinja2 templates which call ``|items``).
      4. Removes the ``index`` key from individual tool_call entries.
    """
    raw_messages = traj.get("messages", [])
    cleaned: list[dict] = []

    for msg in raw_messages:
        role = msg.get("role", "")
        if role not in _VALID_ROLES:
            continue

        out: dict = {"role": role}

        # --- content ---
        content = msg.get("content")
        if content is None:
            out["content"] = ""
        elif isinstance(content, list):
            # Some providers return content as [{"type": "text", "text": "..."}]
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            out["content"] = "\n".join(parts)
        else:
            out["content"] = str(content)

        # --- reasoning_content (GLM-5 CoT) ---
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str) and rc.strip():
            out["reasoning_content"] = rc

        # --- tool_calls ---
        if msg.get("tool_calls"):
            normalised_calls = []
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", {})
                # litellm serialises arguments as a JSON string; templates need a dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {"command": args}
                normalised_calls.append(
                    {
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": func.get("name", "bash"),
                            "arguments": args,
                        },
                    }
                )
            out["tool_calls"] = normalised_calls

        # --- tool_call_id (for tool role) ---
        if role == "tool" and "tool_call_id" in msg:
            out["tool_call_id"] = msg["tool_call_id"]

        cleaned.append(out)

    return cleaned


def get_reasoning_map(traj: dict) -> dict[int, str]:
    """Build a mapping from assistant sequential index → reasoning_content.

    Returns:
        {0: "first assistant CoT", 1: "second assistant CoT", ...}
    """
    mapping: dict[int, str] = {}
    assistant_seq = 0
    for msg in traj.get("messages", []):
        if msg.get("role") == "assistant":
            rc = msg.get("reasoning_content")
            if rc and isinstance(rc, str) and rc.strip():
                mapping[assistant_seq] = rc.strip()
            assistant_seq += 1
    return mapping


# ---------------------------------------------------------------------------
# Style-specific transforms
# ---------------------------------------------------------------------------

def transform_traj_nemotron(traj: dict) -> dict:
    """Nemotron-3-Super style: keep ``reasoning_content`` on every assistant turn.

    The Nemotron chat template enforces that every assistant turn opens with
    ``<think>...</think>``.  When ``reasoning_content`` is present the template
    injects ``<think>{reasoning_content}</think>`` before the content.  When it
    is absent the template inserts an empty ``<think></think>`` automatically.

    Historical think truncation is handled by the template's
    ``truncate_history_thinking`` flag (default True) — we do NOT need to strip
    historical CoT ourselves.
    """
    messages = get_messages(traj)
    reasoning_map = get_reasoning_map(traj)

    assistant_seq = 0
    for msg in messages:
        if msg["role"] == "assistant":
            if assistant_seq in reasoning_map:
                msg["reasoning_content"] = reasoning_map[assistant_seq]
            # If no reasoning, omit the field — template will insert <think></think>
            assistant_seq += 1

    return {"messages": messages}


def transform_traj_qwen35(traj: dict) -> dict:
    """Qwen 3.5 no-think style: strip all ``reasoning_content``.

    For multi-turn agentic SFT, Qwen recommends ``enable_thinking=False``.
    In this mode the template still inserts a minimal ``<think>\\n\\n</think>``
    at the generation prompt, but no CoT content is expected in the training
    messages.  We therefore strip ``reasoning_content`` entirely.
    """
    messages = get_messages(traj)
    # reasoning_content is already not added by get_messages unless present,
    # but explicitly remove it to be safe
    for msg in messages:
        msg.pop("reasoning_content", None)
    return {"messages": messages}


def transform_traj_qwen35_think(traj: dict) -> dict:
    """Qwen 3.5 think style: keep ``reasoning_content`` on every assistant turn.

    The Qwen 3.5 template uses ``last_query_index`` to decide which assistant
    turns get ``<think>`` blocks.  Only turns after the last real user message
    receive the CoT prefix.  Earlier turns have their ``reasoning_content``
    silently ignored by the template.

    We still attach ``reasoning_content`` to every assistant turn so that the
    template can use it where appropriate.  For multi-turn SFT the effective
    CoT is only on the final assistant turn.
    """
    messages = get_messages(traj)
    reasoning_map = get_reasoning_map(traj)

    assistant_seq = 0
    for msg in messages:
        if msg["role"] == "assistant":
            if assistant_seq in reasoning_map:
                msg["reasoning_content"] = reasoning_map[assistant_seq]
            assistant_seq += 1

    return {"messages": messages}


MAP_STYLE_TO_FUNC = {
    "nemotron": transform_traj_nemotron,
    "qwen35": transform_traj_qwen35,
    "qwen35_think": transform_traj_qwen35_think,
}
