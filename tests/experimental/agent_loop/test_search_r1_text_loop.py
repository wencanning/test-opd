import types

import pytest
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.workers.rollout.async_server import TokenOutput


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(tok) for tok in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        rendered = "".join(f"{msg['role']}:{msg['content']}\n" for msg in messages)
        if add_generation_prompt:
            rendered += "assistant:"
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


class FakeRetriever:
    def __init__(self, observation_text):
        self.observation_text = observation_text
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.observation_text


class FakeServerManager:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def generate(self, request_id, prompt_ids, sampling_params, image_data=None):
        self.calls.append(
            {
                "request_id": request_id,
                "prompt_ids": list(prompt_ids),
                "sampling_params": dict(sampling_params),
            }
        )
        return self.responses.pop(0)


def test_search_r1_text_utils_truncate_and_extract():
    from verl.experimental.agent_loop.search_r1_text_utils import (
        extract_answer_text,
        extract_search_query,
        truncate_generation_at_action,
    )

    text = "<think>reason</think><search>penicillin inventor</search> trailing tokens"
    assert truncate_generation_at_action(text) == "<think>reason</think><search>penicillin inventor</search>"
    assert extract_search_query(text) == "penicillin inventor"

    answer_text = "<think>done</think><answer>Alexander Fleming</answer> extra"
    assert truncate_generation_at_action(answer_text) == "<think>done</think><answer>Alexander Fleming</answer>"
    assert extract_answer_text(answer_text) == "Alexander Fleming"


@pytest.mark.asyncio
async def test_search_r1_text_loop_appends_information_with_mask(monkeypatch):
    from verl.experimental.agent_loop.search_r1_text_loop import SearchR1TextAgentLoop
    from verl.experimental.agent_loop.search_r1_text_utils import format_information_observation

    SearchR1TextAgentLoop._class_initialized = False
    raw_observation = "Doc 1 (Title: Penicillin)\nAlexander Fleming discovered penicillin."
    observation = format_information_observation(raw_observation)
    retriever = FakeRetriever(raw_observation)

    monkeypatch.setattr(
        SearchR1TextAgentLoop,
        "_build_retriever_from_config",
        classmethod(lambda cls, config: retriever),
    )

    config = OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {}},
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 512,
                    "response_length": 512,
                    "multi_turn": {
                        "max_assistant_turns": 4,
                        "tool_config_path": None,
                    },
                }
            },
        }
    )

    tokenizer = FakeTokenizer()
    first = "<think>I should verify.</think><search>penicillin inventor</search>"
    second = "<think>The search result is enough.</think><answer>Alexander Fleming</answer>"
    server_manager = FakeServerManager(
        [
            TokenOutput(token_ids=tokenizer.encode(first), log_probs=[-0.1] * len(first)),
            TokenOutput(token_ids=tokenizer.encode(second), log_probs=[-0.2] * len(second)),
        ]
    )
    loop = SearchR1TextAgentLoop(
        trainer_config=types.SimpleNamespace(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )

    result = await loop.run(
        {"temperature": 1.0, "logprobs": 1},
        raw_prompt=[{"role": "user", "content": "Who invented penicillin?"}],
    )

    decoded_response = tokenizer.decode(result.response_ids)
    assert "<search>penicillin inventor</search>" in decoded_response
    assert "<information>Doc 1 (Title: Penicillin)" in decoded_response
    assert "<answer>Alexander Fleming</answer>" in decoded_response
    assert retriever.queries == ["penicillin inventor"]

    first_len = len(tokenizer.encode(first))
    obs_len = len(tokenizer.encode(observation))
    second_len = len(tokenizer.encode(second))
    assert result.response_mask == ([1] * first_len) + ([0] * obs_len) + ([1] * second_len)
    assert result.response_logprobs == ([-0.1] * first_len) + ([0.0] * obs_len) + ([-0.2] * second_len)
    assert result.num_turns == 4
    assert result.extra_fields["extra_info"]["num_tool_calls"] == 1
    assert result.metrics == AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0)


@pytest.mark.asyncio
async def test_search_r1_text_loop_truncates_observation_by_max_obs_length(monkeypatch):
    from verl.experimental.agent_loop.search_r1_text_loop import SearchR1TextAgentLoop
    from verl.experimental.agent_loop.search_r1_text_utils import format_information_observation

    SearchR1TextAgentLoop._class_initialized = False
    raw_observation = "ABCDEFGHIJ"
    max_obs_length = 8
    expected_observation = format_information_observation(raw_observation)
    expected_observation_ids = FakeTokenizer().encode(expected_observation, add_special_tokens=False)[:max_obs_length]
    retriever = FakeRetriever(raw_observation)

    monkeypatch.setattr(
        SearchR1TextAgentLoop,
        "_build_retriever_from_config",
        classmethod(lambda cls, config: retriever),
    )

    config = OmegaConf.create(
        {
            "data": {"apply_chat_template_kwargs": {}, "max_obs_length": max_obs_length},
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 512,
                    "response_length": 512,
                    "multi_turn": {
                        "max_assistant_turns": 4,
                        "tool_config_path": None,
                    },
                }
            },
        }
    )

    tokenizer = FakeTokenizer()
    first = "<search>q</search>"
    second = "<answer>done</answer>"
    server_manager = FakeServerManager(
        [
            TokenOutput(token_ids=tokenizer.encode(first), log_probs=[-0.1] * len(first)),
            TokenOutput(token_ids=tokenizer.encode(second), log_probs=[-0.2] * len(second)),
        ]
    )
    loop = SearchR1TextAgentLoop(
        trainer_config=types.SimpleNamespace(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )

    result = await loop.run(
        {"temperature": 1.0, "logprobs": 1},
        raw_prompt=[{"role": "user", "content": "test"}],
    )

    first_len = len(tokenizer.encode(first))
    second_len = len(tokenizer.encode(second))
    obs_len = len(expected_observation_ids)
    expected_observation_text = tokenizer.decode(expected_observation_ids)

    decoded_response = tokenizer.decode(result.response_ids)
    assert expected_observation_text in decoded_response
    assert result.response_mask == ([1] * first_len) + ([0] * obs_len) + ([1] * second_len)


@pytest.mark.asyncio
async def test_search_r1_text_loop_excludes_observation_from_generation_budget(monkeypatch):
    from verl.experimental.agent_loop.search_r1_text_loop import SearchR1TextAgentLoop
    from verl.experimental.agent_loop.search_r1_text_utils import format_information_observation

    SearchR1TextAgentLoop._class_initialized = False
    raw_observation = "Doc"
    observation = format_information_observation(raw_observation)
    retriever = FakeRetriever(raw_observation)

    monkeypatch.setattr(
        SearchR1TextAgentLoop,
        "_build_retriever_from_config",
        classmethod(lambda cls, config: retriever),
    )

    first = "<search>q</search>"
    second = "<answer>done</answer>"
    generated_budget = len(FakeTokenizer().encode(first)) + len(FakeTokenizer().encode(second))
    config = OmegaConf.create(
        {
            "data": {
                "apply_chat_template_kwargs": {},
                "max_obs_length": 64,
                "max_model_response_length": generated_budget,
            },
            "actor_rollout_ref": {
                "rollout": {
                    "prompt_length": 512,
                    "response_length": generated_budget + 64,
                    "multi_turn": {
                        "max_assistant_turns": 4,
                        "tool_config_path": None,
                    },
                }
            },
        }
    )

    tokenizer = FakeTokenizer()
    server_manager = FakeServerManager(
        [
            TokenOutput(token_ids=tokenizer.encode(first), log_probs=[-0.1] * len(first)),
            TokenOutput(token_ids=tokenizer.encode(second), log_probs=[-0.2] * len(second)),
        ]
    )
    loop = SearchR1TextAgentLoop(
        trainer_config=types.SimpleNamespace(config=config),
        server_manager=server_manager,
        tokenizer=tokenizer,
        processor=None,
    )

    result = await loop.run(
        {"temperature": 1.0, "logprobs": 1},
        raw_prompt=[{"role": "user", "content": "test"}],
    )

    decoded_response = tokenizer.decode(result.response_ids)
    assert observation in decoded_response
    assert "<answer>done</answer>" in decoded_response
