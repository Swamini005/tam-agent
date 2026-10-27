import json
import os
import time

from groq import BadRequestError, RateLimitError

from tam_agent.schema import Decision
from tam_agent.tools import TOOLS

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
TOKEN_LIMIT = int(os.getenv("GROQ_TOKEN_LIMIT", "8000"))

# Groq strict mode requires ALL properties listed in required.
# Pydantic omits Optional / defaulted fields from required, so we patch the schema.
def _strict_schema():
    schema = Decision.model_json_schema()
    # Dereference any $defs so all types are inline
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

    # Allow null for Optional fields
    for field_name in ("account_id", "duplicate_of"):
        prop = schema.get("properties", {}).get(field_name, {})
        if "anyOf" not in prop:
            schema["properties"][field_name] = {"anyOf": [prop, {"type": "null"}]}

    def patch(node):
        """Recursively enforce Groq strict-mode requirements on every object node."""
        if not isinstance(node, dict):
            return node
        node = {k: patch(v) for k, v in node.items()}
        if node.get("type") == "object" or "properties" in node:
            # Every object must have required = all properties and additionalProperties: false
            node["additionalProperties"] = False
            if "properties" in node:
                node["required"] = list(node["properties"].keys())
        # Recurse into anyOf / allOf / oneOf / items
        for key in ("anyOf", "allOf", "oneOf"):
            if key in node:
                node[key] = [patch(sub) for sub in node[key]]
        if "items" in node:
            node["items"] = patch(node["items"])
        return node

    schema = patch(schema)
    return schema


class GroqChat:
    def __init__(self, client, system, prompt):
        self.client = client
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    def _fit(self, extra_tokens):
        budget = (TOKEN_LIMIT - extra_tokens) * 3
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
        """Call Groq, retrying on tool_use_failed and 429 rate-limit errors."""
        wait = 30  # seconds to wait on first 429
        for attempt in range(8):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL,
                    messages=self.messages,
                    max_completion_tokens=max_tokens,
                    **kwargs,
                )
                break
            except RateLimitError:
                if attempt == 7:
                    raise
                print(f"  [rate-limit] waiting {wait}s before retry {attempt + 1}...")
                time.sleep(wait)
                wait = min(wait * 2, 120)  # exponential backoff, cap at 120s
            except BadRequestError as exc:
                if "tool_use_failed" not in str(exc) or attempt == 2:
                    raise

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Groq stopped early: max tokens")
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
        self.messages.append(
            {
                "role": "user",
                "content": f"{instruction}\nDo not call any tools now; reply only with the JSON.",
            }
        )
        schema = _strict_schema()
        self._fit(extra_tokens=3000 + len(json.dumps(schema)) // 3)
        return self._call(
            3000,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "Decision", "schema": schema, "strict": True},
            },
        ).content
