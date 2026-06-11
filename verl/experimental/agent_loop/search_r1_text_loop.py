# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

from .search_r1_text_utils import (
    SearchR1Retriever,
    extract_answer_text,
    extract_search_query,
    format_information_observation,
    load_search_retriever_config,
    truncate_generation_at_action,
)
from .tool_agent_loop import _to_token_id_list

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("search_r1_text_agent")
class SearchR1TextAgentLoop(AgentLoopBase):
    """Text-only Search-R1 rollout loop using <search>/<information>/<answer> tags."""

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        cls.max_assistant_turns = config.actor_rollout_ref.rollout.multi_turn.max_assistant_turns or 8
        cls.retriever = cls._build_retriever_from_config(config)

    @classmethod
    def _build_retriever_from_config(cls, config):
        retriever_config = load_search_retriever_config(config.actor_rollout_ref.rollout.multi_turn.tool_config_path)
        if retriever_config is None:
            return None
        return SearchR1Retriever(retriever_config)

    async def _build_prompt_ids(self, kwargs: dict[str, Any]) -> list[int]:
        raw_prompt_ids = kwargs.get("raw_prompt_ids")
        if raw_prompt_ids is not None:
            return list(raw_prompt_ids)

        messages = list(kwargs["raw_prompt"])
        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            ),
        )
        return _to_token_id_list(prompt_ids)

    async def _search(self, query: str) -> str:
        if self.retriever is None:
            return format_information_observation("Search temporarily unavailable")

        search_result = await self.loop.run_in_executor(None, self.retriever.search, query)
        return format_information_observation(search_result)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        prompt_ids = await self._build_prompt_ids(kwargs)
        running_prompt_ids = list(prompt_ids)

        agent_sampling_params = dict(sampling_params)
        stop_tokens = list(agent_sampling_params.get("stop", []))
        for token in ("</search>", "</answer>"):
            if token not in stop_tokens:
                stop_tokens.append(token)
        agent_sampling_params["stop"] = stop_tokens
        agent_sampling_params.setdefault("include_stop_str_in_output", True)

        request_id = uuid4().hex
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] | None = [] if agent_sampling_params.get("logprobs") else None

        assistant_turns = 0
        successful_searches = 0

        while assistant_turns < self.max_assistant_turns and len(response_ids) < self.response_length:
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=running_prompt_ids,
                sampling_params=agent_sampling_params,
            )
            generated_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            truncated_text = truncate_generation_at_action(generated_text)
            truncated_ids = self.tokenizer.encode(truncated_text, add_special_tokens=False)
            truncated_logprobs = output.log_probs[: len(truncated_ids)] if output.log_probs else None

            running_prompt_ids.extend(truncated_ids)
            response_ids.extend(truncated_ids)
            response_mask.extend([1] * len(truncated_ids))
            if response_logprobs is not None:
                response_logprobs.extend(truncated_logprobs or [0.0] * len(truncated_ids))

            assistant_turns += 1
            if len(response_ids) >= self.response_length:
                break

            if extract_answer_text(truncated_text) is not None:
                break

            search_query = extract_search_query(truncated_text)
            if search_query is None:
                break

            observation_text = await self._search(search_query)
            observation_ids = self.tokenizer.encode(observation_text, add_special_tokens=False)
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break
            observation_ids = observation_ids[:remaining]
            observation_text = self.tokenizer.decode(observation_ids, skip_special_tokens=True)

            running_prompt_ids.extend(observation_ids)
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if response_logprobs is not None:
                response_logprobs.extend([0.0] * len(observation_ids))
            if "<information>" in observation_text:
                successful_searches += 1

        extra_info = {
            "num_tool_calls": successful_searches,
            "num_tool_success": successful_searches,
            "num_turns": 1 + assistant_turns + successful_searches,
        }

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={},
            num_turns=extra_info["num_turns"],
            metrics={},
            extra_fields={"extra_info": extra_info},
        )
