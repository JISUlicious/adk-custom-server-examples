import json
import logging
import re


def parse_qwen_output(raw_output: str) -> dict:
    """
    Worker for Qwen. It now fully parses the JSON content of tool calls
    and extracts the chain-of-thought from <think> blocks.
    """
    logging.debug(">>> Qwen Worker (with thinking) Called <<<")

    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    analysis_content = ""
    think_match = think_pattern.search(raw_output)
    if think_match:
        analysis_content = think_match.group(1).strip()

    parsed_tool_calls = []
    for match in tool_pattern.finditer(raw_output):
        tool_content_str = match.group(1).strip()
        if tool_content_str.startswith("{{") and tool_content_str.endswith("}}"):
            tool_content_str = tool_content_str[1:-1]
        parsed_tool_calls.append(tool_content_str)

    clean_text = think_pattern.sub("", raw_output)
    final_answer = (
        tool_pattern.sub("", clean_text).strip()
        if len(parsed_tool_calls) == 0
        else "\n\n"
    )

    return {
        "analysis": analysis_content,
        "tool_calls": parsed_tool_calls,
        "final_answer": final_answer,
    }


def parse_openai_harmony_output(raw_output: str) -> dict:
    """
    Parses the raw string output from a openai harmony response with a
    regex for handling real-world tool call formats.
    """
    logging.debug(">>> OpenAI Harmony Worker Called <<<")
    if not raw_output.endswith("<|end|>"):
        logging.warning(
            "Raw output does not end with '<|end|>'. This may lead to parsing issues."
        )
        raw_output += "<|end|>"

    # Regex for text channels (this pattern remains correct)
    text_channel_pattern = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>(?P<channel>final|analysis)<\|message\|>(?P<content>.*?)<\|end\|>",
        re.DOTALL,
    )

    # --- THE CORRECTED REGEX FOR TOOL CALLS ---
    # This new pattern correctly matches the structure from your example.
    tool_call_pattern = re.compile(
        r"<\|start\|>assistant<\|channel\|>commentary.*?to=functions\.(?P<name>[\w_]+)"
        r".*?<\|message\|>(?P<args>\{.*?\})<\|call\|>",
        re.DOTALL,
    )

    # Find and process all text channel blocks
    final_answer = None
    analysis_content = ""
    for match in text_channel_pattern.finditer(raw_output):
        channel = match.group("channel")
        content = match.group("content").strip()
        if channel == "analysis":
            # Append to analysis, ensuring we don't overwrite previous content
            analysis_content = analysis_content + "\n" + content
        elif channel == "final":
            # For final, we replace the content if it exists
            final_answer = content

    # Find and process all tool call blocks with the new pattern
    parsed_tool_calls = []
    for match in tool_call_pattern.finditer(raw_output):
        tool_name = match.group("name")
        tool_args_str = match.group("args").strip()
        try:
            arguments = json.loads(tool_args_str)
            parsed_tool_calls.append(
                json.dumps({"name": tool_name, "arguments": arguments})
            )
        except json.JSONDecodeError:
            logging.debug(
                f"Warning: Could not parse JSON arguments for tool '{tool_name}': {tool_args_str}"
            )
            continue

    if not final_answer or len(parsed_tool_calls) != 0:
        final_answer = "\n\n"
    return {
        "analysis": analysis_content,
        "tool_calls": parsed_tool_calls,
        "final_answer": final_answer,
    }


def parse_gemma_output(raw_output: str) -> dict:
    """
    Worker for Gemma family models. Parses Python-like function call syntax
    for tool calls, handling named, positional, multiple, and empty arguments.
    """
    logging.debug(">>> Gemma Worker Called <<<")

    raw_output = raw_output.strip("<end_of_turn>").strip()

    def _parse_gemma_args(arg_string: str) -> dict:
        """A helper function to parse Gemma's complex argument syntax."""
        if not arg_string.strip():
            return {}

        parsed_args = {}
        positional_arg_count = 0
        args = re.split(
            r",(?=(?:[^\'\"`]*[\'\"`][^\'\"`]*[\'\"`])*?[^\'\"`]*$)", arg_string
        )

        for arg in args:
            arg = arg.strip()
            if "=" in arg:
                key, value = arg.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                parsed_args[key] = value
            else:
                key = f"arg{positional_arg_count}"
                value = arg.strip().strip("'\"")
                parsed_args[key] = value
                positional_arg_count += 1
        return parsed_args

    tool_call_pattern = re.compile(
        r"print\((?P<func_name_print>[\w_]+)\s*\((?P<func_args_print>.*?)\)\)|(?P<func_name_direct>[\w_]+)\s*\((?P<func_args_direct>.*?)\)"
    )

    parsed_tool_calls = []

    for match in tool_call_pattern.finditer(raw_output):
        func_name = match.group("func_name_print") or match.group("func_name_direct")
        raw_args = match.group("func_args_print") or match.group("func_args_direct")

        if func_name and raw_args is not None:
            arguments = _parse_gemma_args(raw_args)

            parsed_tool_calls.append(
                json.dumps({"name": func_name, "arguments": arguments})
            )

    final_answer = tool_call_pattern.sub("", raw_output).strip()
    if not parsed_tool_calls:
        final_answer = raw_output.strip()
    else:
        final_answer = "\n\n"

    return {
        "analysis": "",
        "tool_calls": parsed_tool_calls,
        "final_answer": final_answer,
    }


def identify_template_style(prompt_text: str) -> str:
    """
    Correctly identifies the chat template style based only on tokens
    found in the formatted PROMPT text.
    """
    if "<|start|>" in prompt_text and "<|end|>" in prompt_text:
        return "openai_harmony"

    if "<|im_start|>" in prompt_text:
        return "qwen-chatml"

    if "<start_of_turn>" in prompt_text:
        return "gemma"

    if "[INST]" in prompt_text:
        return "llama-instruct"

    return "default"


class LLMOutputParser:
    def __init__(self):
        self._parsers = {
            "qwen-chatml": parse_qwen_output,
            "openai_harmony": parse_openai_harmony_output,
            "gemma": parse_gemma_output,
            "default": lambda text: {"final_answer": text.strip()},
        }

    def get_parser(self, style_name: str):
        return self._parsers.get(style_name, self._parsers["default"])


def identify_template_style(prompt_text: str) -> str:
    """
    Correctly identifies the chat template style based only on tokens
    found in the formatted PROMPT text.
    """
    if "<|start|>" in prompt_text and "<|end|>" in prompt_text:
        return "openai_harmony"

    if "<|im_start|>" in prompt_text:
        return "qwen-chatml"

    if "<start_of_turn>" in prompt_text:
        return "gemma"

    if "[INST]" in prompt_text:
        return "llama-instruct"

    return "default"

# --- Main Parser ---

def parse_chat_template(content: str) -> dict:
    """
    Parses the output from various chat models to identify the template,
    extract thoughts, tool calls, and reformat the output.

    Args:
        content: The raw string output from a chat model.

    Returns:
        A dictionary containing the identified template, extracted parts,
        and a reformatted version if tool calls are present.
    """
    # Base structure for the result
    result = {
        "identified_template": "unknown",
        "has_think": False,
        "has_tool_code": False,
        "raw_think": None,
        "raw_tool_code": None,
        "reformatted_output": None,
        "original_content": content
    }

    style = identify_template_style(content)
    parser_dict = {
            "qwen-chatml": parse_qwen_output,
            "openai_harmony": parse_openai_harmony_output,
            "gemma": parse_gemma_output,
            "default": lambda text: {"final_answer": text.strip()},
        }
    parser = parser_dict.get(style, parser_dict["default"])
    parsed_result = parser(content)
    result.update({
        "identified_template": style,
        "has_think": bool(parsed_result.get("analysis")),
        "has_tool_code": bool(parsed_result.get("tool_calls")),
        "raw_think": parsed_result.get("analysis"),
        "raw_tool_code": parsed_result.get("tool_calls"),
        "reformatted_output": parsed_result if parsed_result.get("tool_calls") else None,
        "original_content": content,
        "final_output": parsed_result.get("final_answer")
    })

    logging.debug(f"parse_chat_template result: {result}")
    return result
