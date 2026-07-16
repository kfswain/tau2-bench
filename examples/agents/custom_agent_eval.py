#!/usr/bin/env python3
"""
Custom agent evaluation example.

This example shows a more complete workflow:
    1. Build components manually (environment, agent, user, orchestrator)
    2. Run a simulation with full control
    3. Inspect the results in detail

This is the "power user" path -- useful for development and debugging.

Usage:
    python examples/agents/custom_agent_eval.py
"""

from typing import Optional
import time

from tau2.agent.base_agent import HalfDuplexAgent
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# =============================================================================
# A slightly more sophisticated agent
# =============================================================================


class VerboseAgent(HalfDuplexAgent[list]):
    """An agent that logs its reasoning before acting.

    This demonstrates how to add custom behavior (logging, pre-processing,
    post-processing) around the core LLM call.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "qwen-32b",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}
        self.call_count = 0
        self._system_messages: list[SystemMessage] = []
        self.total_time = 0.0

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> list[APICompatibleMessage]:
        system_prompt = (
            f"You are a helpful customer service agent.\n\n"
            f"## Policy\n{self.domain_policy}\n\n"
            f"Always follow the policy. Use tools when needed."
        )
        self._system_messages = [SystemMessage(role="system", content=system_prompt)]

        # Replay message history if provided (e.g., for tasks with prior context)
        state: list[APICompatibleMessage] = []
        if message_history:
            state = list(message_history)

        return state

    def generate_next_message(
        self, message: UserMessage, state: list[APICompatibleMessage]
    ) -> tuple[AssistantMessage, list[APICompatibleMessage]]:
        self.call_count += 1
        state.append(message)

        print(
            f"  [VerboseAgent] Turn {self.call_count}: received '{str(message.content)[:80]}...'"
        )

        start_time = time.time()
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=self._system_messages + state,
            **self.llm_args,
        )
        end_time = time.time()
        self.total_time += end_time - start_time

        # Log what the agent decided to do
        print(f"  [VerboseAgent] -> Generated response (raw): '{response}...'")
        if response.tool_calls:
            tool_names = [tc.name for tc in response.tool_calls]
            print(f"  [VerboseAgent] -> Calling tools: {tool_names}")
        else:
            print(
                f"  [VerboseAgent] -> Responding with text: '{str(response.content)[:80]}...'"
            )

        state.append(response)
        return response, state


# =============================================================================
# Build and run manually (no registry needed)
# =============================================================================

from tau2.data_model.simulation import TextRunConfig
from tau2.registry import registry
from tau2.runner import get_tasks, run_tasks
# ... [VerboseAgent class definition remains unchanged] ...
# 1. Define an agent factory function and register it
def create_custom_agent(tools, domain_policy, llm, llm_args, **kwargs):
    return VerboseAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=llm_args,
    )
registry.register_agent_factory(create_custom_agent, "custom_verbose_agent")
if __name__ == "__main__":
    # Load airline domain tasks
    # 14 samples * 10 trials = 140 total requests
    start = 49
    stop = 35
    step = -1
    tasks = get_tasks("airline", task_ids=[str(i) for i in range(start, stop, step)])

    # 2. Initialize the run configuration specifying concurrency settings
    config = TextRunConfig(
        domain="airline",
        agent="custom_verbose_agent",
        user="user_simulator",
        llm_agent="qwen-32b",
        llm_args_agent={
            "api_base": "http://localhost:8000/v1",
            "api_key": "none",
            "custom_llm_provider": "hosted_vllm"
        },
        llm_user="gemini-3.5-flash",
        llm_args_user={
            "api_key": "your key here",
            "custom_llm_provider": "gemini"
        },
        max_concurrency=1000,  # Controls active worker thread limit
        num_trials=10,
        verbose_logs=True,
    )
    # 3. Run via the native framework execution layer
    
    start = time.time()
    results = run_tasks(config, tasks)
    end = time.time()
    print(f"Total execution time: {end - start:.2f} seconds")
    print("Final aggregated results:")
    print(results)
    # The native runner handles console reporting and persistence automatically.
