import sys, json
sys.path.insert(0, '.')
from tam_agent.llm import _strict_schema

schema = _strict_schema()
errors = []

def audit(node, path="$"):
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node:
        props = set(node.get("properties", {}).keys())
        required = set(node.get("required", []))
        missing_req = props - required
        if missing_req:
            errors.append(f"{path}: properties not in required: {missing_req}")
        if node.get("additionalProperties") is not False:
            errors.append(f"{path}: missing additionalProperties=false")
    for key in ("anyOf", "allOf", "oneOf"):
        for i, sub in enumerate(node.get(key, [])):
            audit(sub, f"{path}.{key}[{i}]")
    for k, v in node.get("properties", {}).items():
        audit(v, f"{path}.{k}")
    if "items" in node:
        audit(node["items"], f"{path}.items")

audit(schema)

if errors:
    print("SCHEMA INVALID - Groq will reject:")
    for e in errors:
        print(f"  - {e}")
else:
    print("Schema valid for Groq strict mode: YES")
    print("All properties in required, additionalProperties=false everywhere.")
