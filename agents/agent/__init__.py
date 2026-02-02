from .agent import root_agent
from .agent_utils.register_models import register_models

register_models(
    model_cost={
        "mlx-community/Qwen2.5-7B-Instruct-bf16": {
            "max_tokens": 8192,
            "input_cost_per_token": 0.00000,
            "output_cost_per_token": 0.00000,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
        },
        "Qwen/Qwen3-8B-MLX-bf16": {
            "max_tokens": 32000,
            "input_cost_per_token": 0.00000,
            "output_cost_per_token": 0.00000,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_reasoning": True,
        },
        "inferencerlabs/openai-gpt-oss-20b-MLX-6.5bit": {
            "max_tokens": 32000,
            "input_cost_per_token": 0.00000,
            "output_cost_per_token": 0.00000,
            "litellm_provider": "openai",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_parallel_function_calling": True,
            "supports_reasoning": True,
        },
    }
)

