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
"""
Search-R1 masked SFT dataset.

This dataset consumes a pre-rendered prompt string and a full Search-R1 rollout
response string. Tokens inside <information>...</information> are kept in the
context but masked out from the SFT loss.
"""

from __future__ import annotations

import re

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


INFORMATION_TAG_PATTERN = re.compile(r"<information>.*?</information>", re.DOTALL)


def find_information_spans(text: str) -> list[tuple[int, int]]:
    return [match.span() for match in INFORMATION_TAG_PATTERN.finditer(text)]


class SearchR1SFTDataset:
    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        prompt_key = config.get("prompt_key", "prompt")
        response_key = config.get("response_key", "response")
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = list(parquet_files)
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_length = max_length

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        dataframes = [pd.read_parquet(parquet_file) for parquet_file in self.parquet_files]
        dataframe = pd.concat(dataframes)
        self.prompts = dataframe[self.prompt_key].tolist()
        self.responses = dataframe[self.response_key].tolist()

    def __len__(self):
        return len(self.prompts)

    def _build_response_loss_mask(self, response_text: str, offset_mapping: torch.Tensor) -> torch.Tensor:
        response_loss_mask = torch.ones(offset_mapping.shape[0], dtype=torch.long)

        information_spans = find_information_spans(response_text)
        for idx, (start, end) in enumerate(offset_mapping.tolist()):
            if start == end:
                response_loss_mask[idx] = 0
                continue

            if any(start < span_end and end > span_start for span_start, span_end in information_spans):
                response_loss_mask[idx] = 0

        if response_loss_mask.numel() > 0:
            response_loss_mask[-1] = 0
        return response_loss_mask

    def __getitem__(self, item):
        prompt_text = self.prompts[item]
        response_text = self.responses[item]
        response_with_eos = response_text + self.tokenizer.eos_token

        prompt_ids_output = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        response_ids_output = self.tokenizer(
            response_with_eos,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        response_ids = response_ids_output["input_ids"][0]
        response_attention_mask = response_ids_output["attention_mask"][0]
        response_offsets = response_ids_output["offset_mapping"][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        prompt_loss_mask = torch.zeros(prompt_length, dtype=torch.long)
        response_loss_mask = self._build_response_loss_mask(response_text, response_offsets)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            pad_len = self.max_length - sequence_length
            input_ids = torch.cat(
                (input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=input_ids.dtype))
            )
            attention_mask = torch.cat((attention_mask, torch.zeros((pad_len,), dtype=attention_mask.dtype)))
            loss_mask = torch.cat((loss_mask, torch.zeros((pad_len,), dtype=loss_mask.dtype)))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
                loss_mask = loss_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
            else:
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")

        position_ids = compute_position_id_with_mask(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
