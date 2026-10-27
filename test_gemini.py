import os
from tam_agent.agent import load_dotenv
from tam_agent.llm import get_client_and_config

load_dotenv()

client, provider, model, token_limit = get_client_and_config()
print(f"Provider: {provider}, Model: {model}")

try:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Respond with the word 'OK'"}],
        max_tokens=10
    )
    print("Response SUCCESS:", resp.choices[0].message.content)
except Exception as e:
    print("FAILED:", e)
