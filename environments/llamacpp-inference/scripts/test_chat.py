"""Send a test chat completion request to a running llama.cpp server.

Usage:
    pixi run test-chat                           # default: localhost:8000
    SERVER_URL=http://host:port pixi run test-chat
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    base = os.environ.get("SERVER_URL", "http://localhost:8000")
    url = f"{base}/v1/chat/completions"

    payload = {
        "messages": [
            {"role": "user", "content": "Write a Python function that returns the nth Fibonacci number. Keep it short."}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }

    print(f"POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print()

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as e:
        print(f"ERROR: Could not connect to {url}: {e}", file=sys.stderr)
        print("Is the server running? Start it with: pixi run serve", file=sys.stderr)
        sys.exit(1)

    choice = body["choices"][0]
    print(f"Model: {body.get('model', '(unknown)')}")
    print(f"Finish reason: {choice['finish_reason']}")
    print(f"Usage: {body.get('usage', {})}")
    print()
    print("--- Response ---")
    print(choice["message"]["content"])


if __name__ == "__main__":
    main()
