import json
import logging
import re
from typing import List, Sequence, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_cohere.multi_hop.prompt import multi_hop_prompt_template

"""
    Cohere multi-hop tool allows you to use multiple tools in parallel or 
    sequentially to complete your task. The agent uses a multi hop prompt by Cohere. 
    This prompt is experimental and subject to change.
"""


def create_cohere_multi_hop_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
) -> Runnable:
    multi_hop_prompt = multi_hop_prompt_template.partial(
        tools="\n".join([render_tool_description(t) for t in tools]),
    )
    llm_with_tools = llm.bind(stop=["\nObservation:"])

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_cohere_log_to_str(
                x["intermediate_steps"],
            ),
            input=lambda x: prompt.format_messages(
                input=x["input"], agent_scratchpad=x["intermediate_steps"]
            ),
        )
        | multi_hop_prompt
        | llm_with_tools
        | CohereToolsMultiHopAgentOutputParser()
    )
    return agent


class CohereToolsMultiHopAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    @property
    def _type(self) -> str:
        return "cohere-tools-agent-output-parser"

    def parse_jsonified_tool_use_generation(
        self, tool_use_generation: str, tool_use_prefix: str
    ) -> List[dict]:
        """Parses model-generated jsonified actions.

        Expects input of the form
        "{tool_use_prefix}: ```json\n[{list of jsonified objects}]```"

        outputs parsed list of jsonified objects.
        """

        def _extract_codeblocks_from_md(text: str) -> List[str]:
            return re.findall(r"```([^`]*)```", text)

        raw_generation = re.sub(f"^{tool_use_prefix} ", "", tool_use_generation)
        code_block_sections = _extract_codeblocks_from_md(raw_generation)

        if len(code_block_sections) != 1:  # should have exactly 1 code block
            raise ValueError(f"Action Parsing Failed: {tool_use_generation}")
        # only json allowed:
        assert code_block_sections[0].startswith(
            "json\n"
        ), f"Action Parsing Failed: {tool_use_generation}"

        actions = json.loads(re.sub("^json\n", "", code_block_sections[0]))

        if not isinstance(actions, list):
            raise ValueError(f"Action Parsing Failed: {tool_use_generation}")

        if len(actions):
            if any(
                not isinstance(action, dict) or "tool_name" not in action
                for action in actions
            ):
                raise ValueError(f"Action Parsing Failed: {tool_use_generation}")
        return actions

    def parse_answer_with_prefixes(
        self, completion: str, prefixes: dict[str, str]
    ) -> dict[str, str]:
        """parses string into key-value pairs,
           according to patterns supplied in prefixes. Also strips.

        if inputs are:
            completion = "\nhello: sam\ngoodbye then: paul.",
            prefixes = {"greeting": "hello:", "farewell": "goodbye then:"}

        the expected returned result is:
            {"greeting": "sam", "farewell": "paul."}

        Args:
            completion (str): text to split
            prefixes (Dict[str, str]): a key-value dict of keys and patterns.
            See example above

        Returns:
            Dict[str, str]: parsed result
        """
        # sort out prefixes
        re_pat = "(" + "|".join([re.escape(p) for p in prefixes.values()]) + ")"
        reverse_prefix_map = {v: k for k, v in prefixes.items()}
        split = re.split(re_pat, completion)
        split = split[1:]
        parsed = {}
        for prefix, value in zip(split[::2], split[1::2]):
            if prefix in reverse_prefix_map:  # if the prefix is a match
                if (
                    reverse_prefix_map[prefix] not in parsed
                ):  # first occurrence of a prefix is kept, others discarded
                    parsed[reverse_prefix_map[prefix]] = value.strip()
        return parsed

    def parse_actions(self, generation: str) -> tuple[str, str, List[dict]]:
        """Parse action selections from model output."""
        plan = ""
        actions = generation
        try:
            if "Plan: " in generation or "Reflection: " in generation:
                # Model is trained to output a Plan or Reflection followed by an action.
                # Use regex to extract the plan and action.
                regex = (
                    r"^(Plan|Reflection)\s*\d*\s*:(.*?)"
                    + r"(Action\s*\d*\s*:\s*\d*\s*```json\n.*?```)"
                )
                action_match = re.search(regex, generation, re.DOTALL)
                if not action_match:
                    raise ValueError(
                        f"Failed to parse multihop completion for input: {generation}"
                    )
                plan = action_match.group(2).strip()
                actions = action_match.group(3).strip()
            else:
                # Catch the case where model outputs only an action.
                regex = r"^(Action\s*\d*\s*:\s*\d*\s*```json\n.*?```)"
                action_match = re.search(regex, generation, re.DOTALL)
                if not action_match:
                    raise ValueError(
                        f"Failed to parse multihop completion for input: {generation}"
                    )
                actions = action_match.group(1).strip()
        except Exception as e:
            logging.error(
                f"Failed to parse multihop completion for input: {generation}"
            )
            logging.error(f"Error: {e}")

        parsed_actions = self.parse_jsonified_tool_use_generation(actions, "Action:")
        return generation, plan, parsed_actions

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        # Parse the structured output of the final answer.
        if "Answer: " in text:
            prefix_map = {
                "answer": "Answer:",
                "grounded_answer": "Grounded answer:",
                "relevant_docs": "Relevant Documents:",
                "cited_docs": "Cited Documents:",
            }
            parsed_answer = self.parse_answer_with_prefixes(text, prefix_map)
            return AgentFinish({"output": parsed_answer["answer"]}, text)
        elif any([x in text for x in ["Plan: ", "Reflection: ", "Action: "]]):
            completion, plan, actions = self.parse_actions(text)
            return [
                AgentAction(action["tool_name"], action["parameters"]["query"], text)
                for action in actions
            ]
        else:
            raise ValueError(
                "\nCould not parse generation as it did not contain Plan, Reflection,"
                + f"Action, or Answer. Input: {text}\n\n"
            )


def render_tool_description(tool: BaseTool) -> str:
    """Render the tool in the style of a Python function."""
    function_signature = []
    args_description = []
    for parameter_name, parameter_definition in tool.args.items():
        if "default" in parameter_definition:
            parameter_type = f"Optional[{parameter_definition.get("type", "str")}]"
        else:
            parameter_type = parameter_definition.get("type", "str")
        function_signature += f"{parameter_name}: {parameter_type}"
        args_description += f"{parameter_name} ({parameter_type}): {parameter_definition.get("description", "")}"

    if args_description:
        args_description = """Args:
            """+"\n            ".join(args_description)

    template = PromptTemplate(template_format="f-string").from_template("""```python
    def {TOOL_NAME}({SIGNATURE}) -> List[Dict]:
        \"\"\"
        {DESCRIPTION}
        {ARGS_DESCRIPTION}
        \"\"\"
        pass
    ```""")
    return template.format(
        TOOL_NAME=tool.name,
        DESCRIPTION=tool.description,
        SIGNATURE=", ".join(function_signature),
        ARGS_DESCRIPTION=args_description,
    )


def format_cohere_log_to_str(
    intermediate_steps: List[tuple[AgentAction, str]],
    observation_prefix: str = "<|END_OF_TURN_TOKEN|>\n<|START_OF_TURN_TOKEN|>"
    + "<|SYSTEM_TOKEN|><results>\n",
    llm_prefix: str = "</results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|>"
    + "<|CHATBOT_TOKEN|>",
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    thoughts = ""
    last_log = ""
    # Global result idx to be used for grounded answer citation from any prior action.

    result_idx = 0
    for step_i, (action, observation) in enumerate(intermediate_steps):
        # For parallel tool calls, dont add the action call twice, add all the results.
        if action.log != last_log:
            thoughts += action.log
        last_log = action.log
        if step_i > 0:
            thoughts += "\n"
        observation_str = f"Document {result_idx}\n{observation}"
        result_idx += 1
        thoughts += f"{observation_prefix}{observation_str}{llm_prefix}"

    return thoughts
