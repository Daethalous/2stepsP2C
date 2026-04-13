import json
from openai import OpenAI
import os

client = OpenAI(api_key="sk-xxx")

d = json.load(open('outputs/multi_paper_full_repro_rounds_20260412_211330/llm-detector-evasion/round_1/feature/planning_debug_plan.json'))
messages = d['messages']

try:
    resp = client.post("/chat/completions", cast_to=object, options={"json": {"model": "gpt-4o-mini", "messages": messages}})
    print("Response:", resp)
except Exception as e:
    import traceback
    traceback.print_exc()

