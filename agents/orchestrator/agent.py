import os
import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

# Add agents directory to sys.path to allow importing sibling agents
# Only for top-level Orchesctator agent
agents_dir = Path(__file__).parent.parent
if str(agents_dir) not in sys.path:
    sys.path.insert(0, str(agents_dir))

# Now we can import from sibling agent directories
from worker import root_agent as worker_agent

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
    description="I coordinate sub-agents.",
    instruction="You are the root agent that coordinates sub-agents to handle user requests.",
    sub_agents=[worker_agent],
)
