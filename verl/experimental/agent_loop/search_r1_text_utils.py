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

import json
import re
from dataclasses import dataclass

from omegaconf import OmegaConf

from verl.tools.utils.search_r1_like_utils import perform_single_search_batch

SEARCH_TAG_PATTERN = re.compile(r"<search>(.*?)</search>", re.DOTALL)
ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def truncate_generation_at_action(text: str) -> str:
    """Keep generation up to the first completed search or answer action."""
    candidates = []
    for close_tag in ("</search>", "</answer>"):
        index = text.find(close_tag)
        if index >= 0:
            candidates.append(index + len(close_tag))

    if not candidates:
        return text

    return text[: min(candidates)]


def extract_search_query(text: str) -> str | None:
    matches = SEARCH_TAG_PATTERN.findall(text)
    if not matches:
        return None
    query = matches[-1].strip()
    return query or None


def extract_answer_text(text: str) -> str | None:
    matches = ANSWER_TAG_PATTERN.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    return answer or None


def format_information_observation(observation_text: str) -> str:
    stripped = observation_text.strip()
    return f"\n\n<information>{stripped}</information>\n\n"


@dataclass
class SearchRetrieverConfig:
    retrieval_service_url: str
    topk: int = 3
    timeout: int = 30


def load_search_retriever_config(tool_config_path: str | None) -> SearchRetrieverConfig | None:
    if not tool_config_path:
        return None

    cfg = OmegaConf.load(tool_config_path)
    tools = cfg.get("tools", [])
    for tool in tools:
        tool_cfg = tool.get("config", {})
        retrieval_service_url = tool_cfg.get("retrieval_service_url")
        if retrieval_service_url:
            return SearchRetrieverConfig(
                retrieval_service_url=retrieval_service_url,
                topk=int(tool_cfg.get("topk", 3)),
                timeout=int(tool_cfg.get("timeout", 30)),
            )

    return None


class SearchR1Retriever:
    def __init__(self, config: SearchRetrieverConfig):
        self.config = config

    def search(self, query: str) -> str:
        result_text, _ = perform_single_search_batch(
            retrieval_service_url=self.config.retrieval_service_url,
            query_list=[query],
            topk=self.config.topk,
            concurrent_semaphore=None,
            timeout=self.config.timeout,
        )
        try:
            parsed = json.loads(result_text)
        except json.JSONDecodeError:
            return "Search temporarily unavailable"

        return str(parsed.get("result", "Search temporarily unavailable"))
