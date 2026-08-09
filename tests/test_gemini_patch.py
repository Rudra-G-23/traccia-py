"""
Smoke test for patch_gemini() -- no real API key needed.

Run from inside the traccia-py directory:
    .venv/bin/python3 tests/test_gemini_patch.py
"""

import sys, time, types

sys.path.insert(0, ".")

print("Starting traccia with console exporter...")
import traccia
traccia.start_tracing(service_name="gemini-smoke-test", enable_console_exporter=True)

fake_usage = types.SimpleNamespace(
    total_input_tokens=14,
    total_output_tokens=7,
    total_thought_tokens=0,
)
fake_response = types.SimpleNamespace(
    id="test-001",
    model="gemini-2.0-flash",
    status="completed",
    output_text="The capital of France is Paris.",
    usage=fake_usage,
    agent=None,
)

class FakeInteractions:
    def create(self, **kwargs):
        print("  [fake SDK] model=" + repr(kwargs.get("model")))
        time.sleep(0.05)
        return fake_response

class FakeClient:
    def __init__(self, api_key=None):
        self.interactions = FakeInteractions()

import google
fake_genai = types.ModuleType("google.genai")
fake_genai.Client = FakeClient
google.genai = fake_genai
sys.modules["google.genai"] = fake_genai

fake_resources = types.ModuleType("google.genai.resources")
fake_interactions_mod = types.ModuleType("google.genai.resources.interactions")
fake_interactions_mod.Interactions = FakeInteractions
fake_interactions_mod.AsyncInteractions = None
sys.modules["google.genai.resources"] = fake_resources
sys.modules["google.genai.resources.interactions"] = fake_interactions_mod

# Patch Gemini
from instrumentation.gemini import patch_gemini
result = patch_gemini()
print("patch_gemini() returned:", result, "\n\n")

# Fake call and span on console
print("Fake Gemini call -- span should appear below:")
print("-" * 60)

client = FakeClient(api_key="fake-key")
response = client.interactions.create(
    model="gemini-2.0-flash",
    input="What is the capital of France?",
)

print("-" * 60, "\n\n")
print("Response     :", response)
