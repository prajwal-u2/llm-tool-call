import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

def get_weather(location: str) -> str:
    return json.dumps({"location": location, "temperature": "24 C"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

messages = [{"role": "user", "content": "What is the weather in Minneapolis?"}]

while True:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=tools
    )

    msg = response.choices[0].message
    messages.append(msg)
    print("\n\nAssistant:", msg)

    if not msg.tool_calls:
        break

    tc = msg.tool_calls[0]
    args = json.loads(tc.function.arguments)
    print("\n\nTool call:", tc.function.name, args)

    result = get_weather(**args)
    print("\n\nTool result:", result)

    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    print("\n\n\Tools:", messages)


print("\n\nFinal answer:", messages[-1].content)
print()
