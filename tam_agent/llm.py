import json
import os
import re
import time
import httpx

from tam_agent.schema import Decision
from tam_agent.tools import TOOLS


def _strict_schema():
    """Recursively enforce Groq strict-mode requirements."""
    schema = Decision.model_json_schema()
    defs = schema.pop("$defs", {})

    def deref(node):
        if isinstance(node, dict):
            if "$ref" in node:
                ref_name = node["$ref"].split("/")[-1]
                return deref(defs[ref_name])
            return {k: deref(v) for k, v in node.items()}
        if isinstance(node, list):
            return [deref(i) for i in node]
        return node

    schema = deref(schema)

    for field_name in ("account_id", "duplicate_of"):
        prop = schema.get("properties", {}).get(field_name, {})
        if "anyOf" not in prop:
            schema["properties"][field_name] = {"anyOf": [prop, {"type": "null"}]}

    def patch(node):
        if not isinstance(node, dict):
            return node
        node = {k: patch(v) for k, v in node.items()}
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
        for key in ("anyOf", "allOf", "oneOf"):
            if key in node:
                node[key] = [patch(sub) for sub in node[key]]
        if "items" in node:
            node["items"] = patch(node["items"])
        return node

    return patch(schema)


class GeminiClient:
    """
    Direct HTTP client for Google's OpenAI-compatible endpoint.
    Eliminates SDK path-prefix bugs and prints exact server errors.
    """
    def __init__(self, api_key, model=None):
        self.api_key = api_key
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, model=None, messages=None, tools=None, response_format=None, max_tokens=2000, max_completion_tokens=None, **kwargs):
        req_model = model or self.model
        tokens = max_tokens or max_completion_tokens or 2000
        payload = {
            "model": req_model,
            "messages": messages or [],
            "max_tokens": tokens,
        }
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(self.url, headers=headers, json=payload)
            if resp.status_code != 200:
                # If 404 on gemini-1.5-flash, auto-retry with gemini-2.0-flash
                if resp.status_code == 404 and req_model == "gemini-1.5-flash":
                    payload["model"] = "gemini-2.0-flash"
                    resp2 = client.post(self.url, headers=headers, json=payload)
                    if resp2.status_code == 200:
                        self.model = "gemini-2.0-flash"
                        resp = resp2
                    else:
                        raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
                else:
                    raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")

            data = resp.json()

        class ToolCallFunc:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments if isinstance(arguments, str) else json.dumps(arguments)

        class ToolCallItem:
            def __init__(self, call_id, name, arguments):
                self.id = call_id
                self.type = "function"
                self.function = ToolCallFunc(name, arguments)

        class MessageItem:
            def __init__(self, raw):
                self.content = raw.get("content") or ""
                tcs = raw.get("tool_calls")
                if tcs:
                    self.tool_calls = [
                        ToolCallItem(
                            tc.get("id", f"call_{i}"),
                            tc.get("function", {}).get("name", ""),
                            tc.get("function", {}).get("arguments", "{}")
                        )
                        for i, tc in enumerate(tcs)
                    ]
                else:
                    self.tool_calls = None

        class ChoiceItem:
            def __init__(self, raw):
                self.finish_reason = raw.get("finish_reason")
                self.message = MessageItem(raw.get("message", {}))

        class ResponseItem:
            def __init__(self, raw):
                self.choices = [ChoiceItem(c) for c in raw.get("choices", [])]

        return ResponseItem(data)


def get_client_and_config():
    """
    Auto-detects API key and returns (client, provider, model, token_limit).
    Supports Gemini, Groq, and OpenAI.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if gemini_key:
        provider = "gemini"
        model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        token_limit = int(os.getenv("GEMINI_TOKEN_LIMIT", "1000000"))
        client = GeminiClient(api_key=gemini_key, model=model)
        return client, provider, model, token_limit

    elif groq_key:
        provider = "groq"
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        token_limit = int(os.getenv("GROQ_TOKEN_LIMIT", "8000"))
        from groq import Groq
        client = Groq(max_retries=10)
        return client, provider, model, token_limit

    elif openai_key:
        provider = "openai"
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        token_limit = int(os.getenv("OPENAI_TOKEN_LIMIT", "128000"))
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        return client, provider, model, token_limit

    else:
        raise ValueError("No API key found. Set GEMINI_API_KEY or GROQ_API_KEY in .env.")


class LLMChat:
    def __init__(self, client, provider="groq", model=None, token_limit=None, system="", prompt=""):
        self.client = client
        self.provider = provider
        self.model = model or ("gemini-1.5-flash" if provider == "gemini" else "openai/gpt-oss-120b")
        self.token_limit = token_limit or (1000000 if provider == "gemini" else 8000)
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

    def _fit(self, extra_tokens):
        budget = (self.token_limit - extra_tokens) * 3
        while len(json.dumps(self.messages)) > budget:
            longest = max(
                (m for m in self.messages if m["role"] == "tool"),
                key=lambda m: len(m["content"]),
                default=None,
            )
            if not longest or len(longest["content"]) < 400:
                break
            longest["content"] = (
                longest["content"][: len(longest["content"]) // 2]
                + " ...[cut to fit the token limit]"
            )

    def _call(self, max_tokens, **kwargs):
        wait = 15
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                break
            except Exception as exc:
                err_str = str(exc).lower()
                if "rate" in err_str or "429" in err_str or "quota" in err_str:
                    if attempt == 7:
                        raise
                    print(f"  [rate-limit] waiting {wait}s before retry {attempt + 1}...")
                    time.sleep(wait)
                    wait = min(wait * 2, 60)
                elif "tool_use_failed" in err_str and attempt < 3:
                    time.sleep(2)
                else:
                    raise

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Model stopped early: max tokens")
        message = choice.message
        self.messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                **(
                    {
                        "tool_calls": [
                            {
                                "id": c.id,
                                "type": "function",
                                "function": {
                                    "name": c.function.name,
                                    "arguments": c.function.arguments,
                                },
                            }
                            for c in message.tool_calls
                        ]
                    }
                    if message.tool_calls
                    else {}
                ),
            }
        )
        return message

    def turn(self):
        self._fit(extra_tokens=2000 + len(json.dumps(TOOLS)) // 3)
        calls = []
        for c in self._call(2000, tools=TOOLS).tool_calls or []:
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append((c.id, c.function.name, args))
        return calls

    def send_results(self, results):
        self.messages += [
            {"role": "tool", "tool_call_id": call_id, "content": json.dumps(output)}
            for call_id, _, output in results
        ]

    def final(self, instruction):
        schema = Decision.model_json_schema()
        prompt_text = (
            f"{instruction}\n"
            f"You MUST output a valid JSON object matching this schema exactly:\n"
            f"{json.dumps(schema, indent=2)}\n"
            f"Do not call tools; reply ONLY with the JSON object."
        )
        self.messages.append({"role": "user", "content": prompt_text})

        if self.provider == "groq":
            strict_schema = _strict_schema()
            self._fit(extra_tokens=3000 + len(json.dumps(strict_schema)) // 3)
            return self._call(
                3000,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "Decision", "schema": strict_schema, "strict": True},
                },
            ).content
        else:
            self._fit(extra_tokens=4000)
            try:
                return self._call(
                    3000,
                    response_format={"type": "json_object"},
                ).content
            except Exception:
                return self._call(3000).content


# Backward compatibility alias
GroqChat = LLMChat
