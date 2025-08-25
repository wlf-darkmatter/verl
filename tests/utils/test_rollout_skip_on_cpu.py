# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import verl
from verl.utils.rollout_skip import DataProto, RolloutSkip


def temp_dir():
    # Create a temporary directory
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


def build_generate_fn(gen_bs, n, max_prompt_length, max_response_length):
    len_tokenizer = 1024

    def iterate():
        while True:
            prompt = torch.randint(len_tokenizer, size=(gen_bs, max_prompt_length)).repeat_interleave(n, dim=0)
            generate = torch.randint(len_tokenizer, size=(gen_bs * n, max_response_length))
            data = DataProto.from_dict(tensors={"prompt": prompt, "response": generate})
            yield data

    mock_infer_engine = iterate()

    def fn(batch, **kwargs):
        # Simulate the inference engine returning the next batch
        return next(mock_infer_engine)

    return fn


@pytest.fixture
def mock_rollout_wg():
    default_n = 1
    default_gen_batch_size = 2
    default_max_prompt_length = 16
    default_max_response_length = 16

    config_path = Path(verl.version_folder).joinpath("trainer/config")
    cfg = OmegaConf.load(str(config_path.joinpath("ppo_trainer.yaml")))
    cfg.data = OmegaConf.load(str(config_path.joinpath("data/legacy_data.yaml")))
    cfg.actor_rollout_ref.rollout = OmegaConf.load(config_path.joinpath("rollout/rollout.yaml"))

    temp_dir = tempfile.mkdtemp()

    rollout_wg = MagicMock()

    cfg.trainer.experiment_name = "skip"
    cfg.trainer.project_name = "verl_feat"

    cfg.actor_rollout_ref.rollout.n = default_n
    cfg.actor_rollout_ref.rollout.skip.dump_dir = str(temp_dir)
    cfg.actor_rollout_ref.rollout.skip.dump_step = 1
    cfg.actor_rollout_ref.rollout.skip.enable = True

    cfg.data.gen_batch_size = default_gen_batch_size
    cfg.data.max_prompt_length = default_max_prompt_length
    cfg.data.max_response_length = default_max_response_length

    rollout_wg.generate_sequences = build_generate_fn(
        default_gen_batch_size, default_n, default_max_prompt_length, default_max_response_length
    )

    yield cfg, rollout_wg

    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestRolloutSkip:
    def test_initialization(self, mock_rollout_wg, capsys):
        """Test that RolloutSkip initializes correctly"""
        config, rollout_wg = mock_rollout_wg

        skip = RolloutSkip(config)

        assert skip.n == config.actor_rollout_ref.rollout.n
        assert skip.gbs == config.data.gen_batch_size
        assert skip.prompt_length == config.data.max_prompt_length
        assert skip.response_length == config.data.max_response_length
        assert skip.do_compress == config.actor_rollout_ref.rollout.skip.compress
        assert skip.strict_mode == config.actor_rollout_ref.rollout.skip.strict_mode
        assert skip.is_enable
        assert str(skip.specify_dumped_dir).startswith(config.actor_rollout_ref.rollout.skip.dump_dir)

        assert not skip.is_dump_step
        assert not skip.is_activate
        skip.wrap_generate_sequences(rollout_wg)

        assert skip.is_dump_step
        assert skip.is_activate

        assert skip._rollout_wg == rollout_wg

        captured = capsys.readouterr()
        assert "Successfully patched" in captured.out

    def test_generate_without_wrap(self, mock_rollout_wg):
        """Test that generate_sequences works without wrapping"""

        config, rollout_wg = mock_rollout_wg
        _ = RolloutSkip(config)

        _result = rollout_wg.generate_sequences(MagicMock())
        for _ in range(10):
            result = rollout_wg.generate_sequences(MagicMock())
            assert isinstance(result, DataProto)
            # * make sure the data is different
            assert torch.abs(_result.batch["prompt"] - result.batch["prompt"]).sum() > 0
            assert torch.abs(_result.batch["response"] - result.batch["response"]).sum() > 0
            _result = result

    def test_generate_with_wrap_nostrict(self, mock_rollout_wg, capsys):
        """Test that generate_sequences works without wrapping"""

        config, rollout_wg = mock_rollout_wg
        skip = RolloutSkip(config)
        skip.strict_mode = False
        skip.wrap_generate_sequences(rollout_wg)

        _result = rollout_wg.generate_sequences(MagicMock())

        for _ in range(10):
            result = rollout_wg.generate_sequences(MagicMock())
            assert isinstance(result, DataProto)
            # * make sure the data is different
            assert torch.allclose(_result.batch["prompt"], result.batch["prompt"])
            assert torch.allclose(_result.batch["response"], result.batch["response"])

            captured = capsys.readouterr()
            assert "Successfully load pre-generated data from" in captured.out
            _result = result

    def test_dump_nostrict(self, mock_rollout_wg, capsys):
        config, rollout_wg = mock_rollout_wg
        skip = RolloutSkip(config)
        skip.strict_mode = False
        skip.wrap_generate_sequences(rollout_wg)

        result = rollout_wg.generate_sequences(MagicMock())
        # * check if dump is OK
        assert skip.get_path_dump().exists()
        captured = capsys.readouterr()
        assert "Successfully dump data in" in captured.out
        # * get file size, estimate file size
        file_size = skip.get_path_dump().stat().st_size
        if not skip.do_compress:
            est_file_size = (
                (skip.prompt_length + skip.response_length) * skip.gbs * skip.n * result.batch["prompt"].dtype.itemsize
            )
            assert file_size >= est_file_size, "Dumped file size is smaller than expected"
