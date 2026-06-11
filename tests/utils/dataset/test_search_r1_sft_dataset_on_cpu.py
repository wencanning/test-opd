import json

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text, return_tensors=None, add_special_tokens=False, return_offsets_mapping=False):
        input_ids = [ord(ch) for ch in text]
        attention_mask = [1] * len(input_ids)
        output = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
        if return_offsets_mapping:
            offsets = [[i, i + 1] for i in range(len(text))]
            output["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return output

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return "".join(chr(tok) for tok in token_ids if tok != self.pad_token_id)


def test_search_r1_sft_dataset_masks_information_tokens(tmp_path):
    from verl.utils.dataset.search_r1_sft_dataset import SearchR1SFTDataset, find_information_spans

    prompt = "system\nuser question\nassistant\n"
    response = (
        "<think>need search</think><search>q</search>\n\n"
        "<information>Doc 1</information>\n\n"
        "<think>done</think><answer>Yes</answer>"
    )
    data_file = tmp_path / "search_r1_sft.parquet"
    pd.DataFrame(
        [
            {
                "prompt": prompt,
                "response": response,
            }
        ]
    ).to_parquet(data_file, index=False)

    dataset = SearchR1SFTDataset(
        parquet_files=str(data_file),
        tokenizer=FakeTokenizer(),
        config=OmegaConf.create(
            {
                "prompt_key": "prompt",
                "response_key": "response",
                "max_length": 512,
                "truncation": "error",
            }
        ),
    )

    item = dataset[0]
    loss_mask = item["loss_mask"]
    input_ids = item["input_ids"]
    attention_mask = item["attention_mask"]

    valid_length = int(attention_mask.sum().item())
    decoded_masked = "".join(chr(int(tok)) for tok in input_ids[:valid_length][loss_mask[:valid_length] == 1])
    decoded_unmasked = "".join(chr(int(tok)) for tok in input_ids[:valid_length][loss_mask[:valid_length] == 0])

    assert "<search>q</search>" in decoded_masked
    assert "<answer>Yes</answer>" in decoded_masked
    assert "<information>Doc 1</information>" not in decoded_masked
    assert "<information>Doc 1</information>" in decoded_unmasked
    assert loss_mask[valid_length - 1].item() == 0
    assert torch.all(loss_mask[: len(prompt)] == 0)
    spans = find_information_spans(response)
    assert len(spans) == 1
    assert response[spans[0][0] : spans[0][1]] == "<information>Doc 1</information>"


def test_search_r1_sft_dataset_masks_multiple_information_spans(tmp_path):
    from verl.utils.dataset.search_r1_sft_dataset import SearchR1SFTDataset

    prompt = "prompt\nassistant\n"
    response = (
        "<think>a</think><search>q1</search>\n\n"
        "<information>Doc A</information>\n\n"
        "<think>b</think><search>q2</search>\n\n"
        "<information>Doc B</information>\n\n"
        "<think>c</think><answer>Done</answer>"
    )
    data_file = tmp_path / "search_r1_multi.parquet"
    pd.DataFrame([{"prompt": prompt, "response": response}]).to_parquet(data_file, index=False)

    dataset = SearchR1SFTDataset(
        parquet_files=str(data_file),
        tokenizer=FakeTokenizer(),
        config=OmegaConf.create({"prompt_key": "prompt", "response_key": "response", "max_length": 512}),
    )
    item = dataset[0]
    valid_length = int(item["attention_mask"].sum().item())
    decoded_masked = "".join(chr(int(tok)) for tok in item["input_ids"][:valid_length][item["loss_mask"][:valid_length] == 1])
    decoded_unmasked = "".join(
        chr(int(tok)) for tok in item["input_ids"][:valid_length][item["loss_mask"][:valid_length] == 0]
    )

    assert decoded_masked.count("<search>") == 2
    assert "<answer>Done</answer>" in decoded_masked
    assert "<information>Doc A</information>" not in decoded_masked
    assert "<information>Doc B</information>" not in decoded_masked
    assert "<information>Doc A</information>" in decoded_unmasked
    assert "<information>Doc B</information>" in decoded_unmasked


def test_search_r1_sft_dataset_right_truncation_preserves_mask_alignment(tmp_path):
    from verl.utils.dataset.search_r1_sft_dataset import SearchR1SFTDataset

    prompt = "prompt\nassistant\n"
    response = "<think>x</think><information>MASKME</information><answer>OK</answer>"
    data_file = tmp_path / "search_r1_trunc.parquet"
    pd.DataFrame([{"prompt": prompt, "response": response}]).to_parquet(data_file, index=False)

    dataset = SearchR1SFTDataset(
        parquet_files=str(data_file),
        tokenizer=FakeTokenizer(),
        config=OmegaConf.create(
            {
                "prompt_key": "prompt",
                "response_key": "response",
                "max_length": len(prompt) + len("<think>x</think><information>MASKME</information>"),
                "truncation": "right",
            }
        ),
    )

    item = dataset[0]
    valid_length = int(item["attention_mask"].sum().item())
    masked_text = "".join(chr(int(tok)) for tok in item["input_ids"][:valid_length][item["loss_mask"][:valid_length] == 1])
    unmasked_text = "".join(
        chr(int(tok)) for tok in item["input_ids"][:valid_length][item["loss_mask"][:valid_length] == 0]
    )

    assert "MASKME" not in masked_text
    assert "<information>MASKME</information>" in unmasked_text
    assert "<answer>" not in masked_text
    assert masked_text.startswith("<think>")


def test_create_sft_dataset_selects_search_r1_dataset(tmp_path):
    from verl.trainer.fsdp_sft_trainer import create_sft_dataset
    from verl.utils.dataset.search_r1_sft_dataset import SearchR1SFTDataset

    data_file = tmp_path / "search_r1_sft.parquet"
    pd.DataFrame([{"prompt": "p", "response": "r"}]).to_parquet(data_file, index=False)

    dataset = create_sft_dataset(
        str(data_file),
        OmegaConf.create(
            {
                "prompt_key": "prompt",
                "response_key": "response",
                "max_length": 32,
                "search_r1_masked": {"enable": True},
            }
        ),
        FakeTokenizer(),
    )

    assert isinstance(dataset, SearchR1SFTDataset)


def test_preprocess_search_r1_sft_rollouts_build_and_split_rows(tmp_path):
    from examples.data_preprocess.preprocess_search_r1_sft_rollouts import build_sft_row, load_rows, split_rows

    accepted = {
        "sample_id": "id-1",
        "data_source": "hotpotqa",
        "question": "Q1",
        "rendered_prompt": "PROMPT1",
        "response": "<think>a</think><answer>Yes</answer>",
        "ground_truth": {"target": ["yes"]},
        "answer_correct": True,
        "quality_score": 10,
        "quality_pass": True,
        "search_turn_count": 1,
        "information_turn_count": 1,
    }
    fallback = {
        "sample_id": "id-2",
        "data_source": "nq",
        "question": "Q2",
        "rendered_prompt": "PROMPT2",
        "trajectory_text": "<think>b</think><answer>No</answer>",
        "ground_truth": {"target": ["no"]},
        "answer_correct": False,
        "quality_score": 5,
        "quality_pass": False,
        "search_turn_count": 0,
        "information_turn_count": 0,
    }
    invalid = {"sample_id": "id-3", "response": "missing prompt"}

    assert build_sft_row(accepted)["response"] == accepted["response"]
    assert build_sft_row(fallback)["response"] == fallback["trajectory_text"]
    assert build_sft_row(invalid) is None

    jsonl = tmp_path / "accepted.jsonl"
    with jsonl.open("w") as f:
        f.write(json.dumps(accepted) + "\n")
        f.write(json.dumps(fallback) + "\n")
        f.write(json.dumps(invalid) + "\n")

    rows = load_rows(jsonl)
    assert len(rows) == 2
    train_rows, val_rows = split_rows(rows, val_ratio=0.5, seed=123)
    assert len(train_rows) == 1
    assert len(val_rows) == 1
    assert {row["sample_id"] for row in train_rows + val_rows} == {"id-1", "id-2"}
