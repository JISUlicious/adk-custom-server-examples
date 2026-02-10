import os

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

model_name = os.getenv("OPENROUTER_MODEL_GLM_4_5_AIR")
api_base = os.getenv("OPENROUTER_API_BASE")
api_key = os.getenv("OPENROUTER_API_KEY")

agent_model = LiteLlm(
    model=model_name,
    api_base=api_base,
    api_key=api_key,
    stream=True,
)

root_agent = LlmAgent(
    output_key="result",
    name="root_agent",
    model=agent_model,
    description="Worker agent for orchestrator.",
    instruction="You are the root agent that takes tasks from orchestrator agent.",
)
