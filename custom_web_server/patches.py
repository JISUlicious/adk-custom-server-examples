"""
Patches for ADK bugs/missing features.

These patches fix issues in ADK that haven't been resolved upstream yet.
"""

import logging

logger = logging.getLogger(__name__)


def patch_litellm_reasoning_extraction():
    """
    Patch ADK's LiteLLM integration to properly extract reasoning from streaming deltas.

    ADK's _extract_reasoning_value only looks for 'reasoning_content' attribute,
    but LiteLLM's ChoiceDelta uses 'reasoning' and 'reasoning_details'.

    This patch extends the function to also check these attributes.
    """
    try:
        from google.adk.models import lite_llm

        # Store original function
        original_extract = lite_llm._extract_reasoning_value

        def patched_extract_reasoning_value(message):
            """Extended reasoning extraction that handles LiteLLM's streaming format."""
            # First try the original logic
            result = original_extract(message)
            if result:
                return result

            # Check for 'reasoning' attribute (used in streaming deltas by OpenRouter, etc.)
            if hasattr(message, 'reasoning') and message.reasoning:
                return message.reasoning

            # Check for 'reasoning_content' (already checked by original but be thorough)
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                return message.reasoning_content

            # Check for 'reasoning_details' attribute
            if hasattr(message, 'reasoning_details') and message.reasoning_details:
                return message.reasoning_details

            # Check dict format
            if isinstance(message, dict):
                if message.get('reasoning'):
                    return message.get('reasoning')
                if message.get('reasoning_content'):
                    return message.get('reasoning_content')
                if message.get('reasoning_details'):
                    return message.get('reasoning_details')

            return None

        # Apply patch
        lite_llm._extract_reasoning_value = patched_extract_reasoning_value
        print("[PATCH] Applied ADK reasoning extraction patch")
        logger.info("Applied ADK reasoning extraction patch")
        return True

    except Exception as e:
        logger.warning("Failed to apply ADK reasoning patch: %s", e)
        return False


def patch_litellm_streaming_handler():
    """
    Patch LiteLLM's CustomStreamWrapper to handle reasoning-only chunks.

    LiteLLM's streaming handler drops reasoning-only chunks because:
    1. handle_openai_chat_completion_chunk only extracts delta.content, not delta.reasoning
    2. is_chunk_non_empty checks for reasoning_content but providers send reasoning

    This patch fixes both issues to preserve reasoning content in streaming.
    """
    try:
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

        # Patch 1: Fix is_chunk_non_empty to check for 'reasoning' attribute
        original_is_chunk_non_empty = CustomStreamWrapper.is_chunk_non_empty

        def patched_is_chunk_non_empty(self, completion_obj, model_response, response_obj):
            # First check original logic
            result = original_is_chunk_non_empty(self, completion_obj, model_response, response_obj)
            if result:
                return True

            # Also check for 'reasoning' in the original_chunk delta
            original_chunk = response_obj.get("original_chunk")
            if original_chunk and hasattr(original_chunk, 'choices') and original_chunk.choices:
                delta = getattr(original_chunk.choices[0], 'delta', None)
                if delta:
                    # Check for reasoning (used by OpenRouter, etc.)
                    if hasattr(delta, 'reasoning') and delta.reasoning:
                        return True
                    # Check for reasoning_details
                    if hasattr(delta, 'reasoning_details') and delta.reasoning_details:
                        return True

            return False

        CustomStreamWrapper.is_chunk_non_empty = patched_is_chunk_non_empty
        print("[PATCH] Applied is_chunk_non_empty patch for reasoning")

        # Patch 2: Copy reasoning to reasoning_content for compatibility
        # This ensures downstream code that checks reasoning_content will find it
        original_chunk_creator = CustomStreamWrapper.chunk_creator

        def patched_chunk_creator(self, chunk):
            # Call original chunk creator
            result = original_chunk_creator(self, chunk)

            # If result is valid and has choices with delta, check for reasoning
            if result and hasattr(result, 'choices') and result.choices:
                delta = getattr(result.choices[0], 'delta', None)
                if delta:
                    # Copy reasoning to reasoning_content if present
                    reasoning = getattr(delta, 'reasoning', None)
                    if reasoning and not getattr(delta, 'reasoning_content', None):
                        try:
                            delta.reasoning_content = reasoning
                        except AttributeError:
                            # Delta might be immutable, try setting via dict
                            if hasattr(delta, '__dict__'):
                                delta.__dict__['reasoning_content'] = reasoning

            return result

        CustomStreamWrapper.chunk_creator = patched_chunk_creator
        print("[PATCH] Applied chunk_creator patch for reasoning")

    except Exception as e:
        print(f"[PATCH] Could not patch CustomStreamWrapper: {e}")
        import traceback
        traceback.print_exc()

    # Also try to set LiteLLM to pass through unknown fields
    try:
        import litellm
        # Enable passing through additional response fields
        if hasattr(litellm, 'drop_params'):
            litellm.drop_params = False
        print("[PATCH] Set litellm.drop_params = False")
    except Exception as e:
        print(f"[PATCH] Could not set drop_params: {e}")


def apply_all_patches():
    """Apply all patches."""
    patch_litellm_reasoning_extraction()
    patch_litellm_streaming_handler()
