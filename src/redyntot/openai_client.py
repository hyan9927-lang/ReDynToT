"""A configurable chat-completions transport with explicit generated-token log probabilities."""
import json as jsonlib
import math
import os
import urllib.error
import urllib.request


class ChatResponse:
    def __init__(self, data):
        self.data = data

    def json(self):
        return self.data

    def raise_for_status(self):
        return None


class ChatClient:
    def __init__(self, model, timeout=180):
        self.model = model
        self.url = os.environ.get("REDYNTOT_CHAT_URL") or os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/') + '/chat/completions'
        if not self.url:
            raise ValueError("Set REDYNTOT_CHAT_URL to the complete chat-completions endpoint")
        self.timeout = timeout
        self.calls = []
        self.failures = []

    def post(self, url, json):
        payload = dict(json)
        payload["model"] = self.model
        headers = {"Content-Type": "application/json"}
        credential = os.environ.get("REDYNTOT_API_KEY") or os.environ.get('OPENAI_API_KEY')
        if credential:
            headers["Authorization"] = "Bearer " + credential
        request = urllib.request.Request(self.url, data=jsonlib.dumps(payload).encode("utf-8"), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = jsonlib.load(response)
            choice = data["choices"][0]
            if choice.get("finish_reason") == "length":
                raise ValueError("Model output was truncated")
            content = choice.get("logprobs", {}).get("content")
            if not isinstance(content, list) or not content:
                raise ValueError("The selected provider must return generated-token logprobs.content")
            if not all(isinstance(x.get("logprob"), (int, float)) and math.isfinite(x["logprob"]) for x in content):
                raise ValueError("Missing or nonfinite token log probability")
            self.calls.append({"requested_model": self.model, "returned_model": data.get("model"), "usage": data.get("usage", {}), "response_id": data.get("id")})
            return ChatResponse(data)
        except Exception as error:
            self.failures.append(type(error).__name__)
            raise

    def complete(self, system, user, max_tokens=1024, stop=None):
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}], "temperature": 0, "max_tokens": max_tokens, "logprobs": True}
        if stop is not None:
            payload["stop"] = stop
        return self.post(self.url, json=payload).json()["choices"][0]
