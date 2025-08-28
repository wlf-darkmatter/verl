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
import logging
import lzma  # * 压缩率最高，耗时最多，内存占用最少
import pickle
from enum import Enum
from pathlib import Path

import torch

from verl.protocol import DataProto


class PostDumpAction(Enum):
    REPEAT = "repeat"  # 循环使用所有dump的样本
    REPEAT_LAST = "repeat_last"  # 重复使用最后一个样本
    ROLLOUT = "rollout"  # 恢复正常推理模式
    ROLLOUT_WITH_DUMP = "rollout_with_dump"
    EXIT = "exit"  # 退出程序


class RolloutSkip:
    """
    RolloutSkip skips sequence generation during rollout by attempting to load previously dumped data.
    If no dumped data is found, it generates new sequences and saves them to disk.

    Args:
        config: The configuration object containing rollout settings.
        rollout_wg: The worker group that handles the rollout process.

    Note:
        Whenever any of the following parameters differ from previous runs—trainer.experiment_name,
        trainer.project_name, rollout.n, or rollout.gen_batch_size—new sequences will be generated
        and saved under different filenames.


    行为模式有2种:

    A: 如果未超出设置的 dump 步长:

        I: 发现对应步数的 dump 内容存在, 则使用 dump 内容，跳过生成 ✅
        II: 发现对应步数的 dump 内容不存在, 则进行推理，并进行 dump ✅

    B. 如果超出设置的 dump 步长，

        按照 post_dump_action 的设置行为处理
        # circular, replicate, exit, continues
            1. "REPEAT": 循环使用之前的 dump 内容
            2. "REPEAT_LAST": 重复使用最后的 dump 内容
            3. "EXIT": 退出程序 ✅
            4. "ROLLOUT": 恢复正常的 rollout，不使用 dump，也不进行 dump ✅
            5. "ROLLOUT_WITH_DUMP": 恢复正常的 rollout，进行 dump ✅


    功能增强：
        1. 默认支持 strict_mode，开启后会对比 prompt 的一致性，不一致则会对 new_batch 的 prompt 和 answer 进行修改
        2. 默认支持 compress，开启后会对 dump 的内容进行压缩，节省存储空间
        3. 添加输入匹配检查 ✅

    """

    print_mark = "[RolloutSkip()] "
    name_metadata = "metadate.json"

    def __init__(self, config):
        self.rollout_config = config.actor_rollout_ref.rollout
        self.skip_config = self.rollout_config.skip
        self.exp_name = config.trainer.get("experiment_name", "")
        self.project_name = config.trainer.get("project_name", "")
        self.n = int(self.rollout_config.get("n", 0))
        self.gbs = int(config.data.get("gen_batch_size", config.data.get("train_batch_size", 0)))
        self.response_length = config.data.max_response_length
        self.prompt_length = config.data.max_prompt_length

        self._rollout_wg = None
        self._new_batch = None
        self._curr_step: int = 0

        self.strict_mode = self.skip_config.get("strict_mode", True)
        self.do_compress = self.skip_config.get("compress", True)
        self.dump_step = self.skip_config.get("dump_step", 1)
        self.post_dump_action = self.skip_config.get("post_dump_action", PostDumpAction.REPEAT)
        self.post_dump_action = PostDumpAction(self.post_dump_action)

        self._create_dump_path()
        self._flag_record = False

    @property
    def is_enable(self) -> bool:
        return self.skip_config.get("enable", False)

    @property
    def is_activate(self) -> bool:
        """
        If RolloutSkip is enabled and the rollout worker group is set, it is considered active.
        """
        return self.is_enable and self._rollout_wg is not None

    @property
    def curr_path_dump(self) -> Path:
        return self.dumped_dir.joinpath(f"genstep_{self._curr_step:06d}.pkl").absolute()

    @property
    def curr_step(self) -> int:
        return self._curr_step

    @property
    def is_dump_step(self) -> bool:
        """
        Determine if the current step is a dump step based on the configured dump interval.
        """
        return self.is_activate and self._curr_step <= self.dump_step

    def _create_dump_path(self):
        """
        Create the directory for dumping rollout data if it doesn't exist.
        Warn if the directory is within Ray's temporary session directory.
        """

        dumped_dir = Path(self.skip_config.get("dump_dir", "/tmp/verl/rollout_dump"))
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )

        self.dumped_dir = dumped_dir.joinpath(sub_dir)
        self.dumped_dir.mkdir(parents=True, exist_ok=True)

        tmp_ray = "/tmp/ray/session"

        # Check if path is in Ray temporary directory
        if str(self.dumped_dir.absolute()).startswith(tmp_ray):
            print(
                f"{self.print_mark}\033[33mWarning: \nUsing dump path ",
                f"'{self.dumped_dir.absolute()}' is not recommended ",
                f"as it's located in {tmp_ray}*\033[0m",
                flush=True,
            )
        print(
            f"{self.print_mark}Rollout skip dump path set to: ",
            str(self.dumped_dir.absolute()),
            flush=True,
        )

    def record(self, new_batch: DataProto, *args, **kwargs):
        """Record the current training step based on the new batch.

        Args:
            new_batch (DataProto): The new batch of data being processed.
        """
        if self._rollout_wg is None:
            return
        if self._flag_record is False:
            # * 保证一个 record 对应一次 skip
            self._flag_record = True
            self._new_batch = new_batch
            self._curr_step += 1
        else:
            print(f"{self.print_mark}Warning, duplicate record new_batch.", flush=True)

    def wrap_generate_sequences(self, rollout_wg):
        self._rollout_wg = rollout_wg
        try:
            self._rollout_wg.generate_sequences = wrap_generate_sequences(self, self._rollout_wg)
            print(
                f"{self.print_mark}\033[32mSuccessfully patched `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"{self.print_mark}\033[31mFailed to patch `actor_rollout_wg.generate_sequences()`.\033[0m",
                flush=True,
            ) from e

    def try_load(self, input_batch: DataProto = None):
        dumped_batch = None
        if self.curr_path_dump.exists():
            try:
                # * Load
                data_dict = pickle.loads(self.curr_path_dump.read_bytes())
                dataproto_decompress(data_dict)

                dumped_batch = data_dict["gen_batch"]

                print(
                    f"{self.print_mark}\033[32mSuccessfully load pre-generated data from {self.curr_path_dump}.\033[0m",
                    flush=True,
                )
                # * Check if new_batch's prompt matches dumped_batch's prompt
                if self.check_data(input_batch, dumped_batch, verbose=True):
                    self.replace_new_batch(dumped_batch)

            except Exception as e:
                print(
                    f"{self.print_mark}\033[31mFailed to load pre-generated data from {self.curr_path_dump}.\033[0m",
                    flush=True,
                )
                logging.exception(e)

        else:
            print(
                f"{self.print_mark}\033[33mNo dumped data found at gen_step {self._curr_step} ",
                f"from {self.curr_path_dump}. The trainer will generate and dump the data for this gen_step.\033[0m",
                flush=True,
            )

        return dumped_batch

    def dump(self, outputs: DataProto):
        data_dump = {
            "new_batch": self._new_batch,
            "gen_batch": outputs,
            "compressed": [],
        }

        try:
            info_compress = ""
            if self.do_compress:
                data_dump["compressed"] = ["gen_batch", "new_batch"]
                size_zip, size_data = dataproto_compress(data_dump)
                if size_data != 0:
                    ratio = size_zip / size_data
                    info_compress = (
                        f"Compressed Ratio: {ratio:.1%} ({size_data / 1024**2:.3f}MB -> {size_zip / 1024**2:.3f}MB)"
                    )

            with open(str(self.curr_path_dump), "wb") as f:
                pickle.dump(data_dump, f)

            print(
                f"{self.print_mark}\033[32mSuccessfully dump data in {self.curr_path_dump}\033[0m",
                info_compress,
                flush=True,
            )

        except Exception as e:
            print(
                f"{self.print_mark}\033[31mFailed to dump data in {self.curr_path_dump}: {e}\033[0m",
                flush=True,
            )

    def replace_new_batch(self, dumped_batch: DataProto):
        breakpoint()
        pass

    @torch.no_grad()
    def check_data(
        self,
        input_batch: DataProto,
        dumped_batch: DataProto,
        verbose=False,
        strict=False,
    ):
        """
        Compare the prompts in the current batch with those in the dumped batch to ensure they match.
        """
        strict = strict or self.strict_mode

        len_prompt = self.prompt_length
        size_input = input_batch.batch["input_ids"].shape
        check_prompt = True
        info = ""
        # * strong check
        if size_input[1] != len_prompt:
            info += f"Input prompt length({size_input[1]}) mismatch with max_prompt_length({len_prompt}). "
            check_prompt = False

        if size_input[0] != (self.gbs * self.n):
            info += f"Input prompt bs({size_input[0]}) mismatch with gen_batch_size({self.gbs}) * n({self.n}). "
            check_prompt = False

        if strict and check_prompt:
            # * weak check
            ids_diff = input_batch.batch["input_ids"][:, :len_prompt] == dumped_batch.batch["input_ids"][:, :len_prompt]
            if not ids_diff.all().item():
                check_prompt = False
                info += "Prompt of `input_batch` != prompt of `dumped_batch`. "
                info += f"Element-wise Similarity : {ids_diff.sum() / ids_diff.numel():.2%}"

        if verbose and check_prompt is False:
            print(
                f"{self.print_mark}\033[33mWarning: Prompt mismatch detected at gen_step {self.curr_step}: "
                f"{info}"
                "Possible dataset change or unseeded shuffle.\033[0m",
                flush=True,
            )

        return check_prompt


def wrap_generate_sequences(rolloutskip: RolloutSkip, rollout_wg):
    generate_sequences = rollout_wg.generate_sequences

    def rollout_skip_wrap_fn(batch, **kwargs):
        if rolloutskip.is_dump_step:
            dumped_batch = rolloutskip.try_load(input_batch=batch)

            if dumped_batch is None:
                # * 1. Generation
                dumped_batch = generate_sequences(batch, **kwargs)
                # * 2. Dump
                rolloutskip.dump(dumped_batch)
        else:
            rolloutskip.replace_new_batch(batch)

            if rolloutskip.post_dump_action == PostDumpAction.REPEAT:
                rolloutskip.replace_new_batch()  # todo
            elif rolloutskip.post_dump_action == PostDumpAction.REPEAT_LAST:
                rolloutskip.replace_new_batch()  # todo

            elif rolloutskip.post_dump_action == PostDumpAction.ROLLOUT:
                dumped_batch = generate_sequences(batch, **kwargs)
            elif rolloutskip.post_dump_action == PostDumpAction.ROLLOUT_WITH_DUMP:
                dumped_batch = generate_sequences(batch, **kwargs)
                rolloutskip.dump(dumped_batch)
            elif rolloutskip.post_dump_action == PostDumpAction.EXIT:
                exit(0)

        # clean
        rolloutskip._flag_record = False
        return dumped_batch

    return rollout_skip_wrap_fn


def dataproto_compress(dict_data: dict) -> dict[str, DataProto]:
    key_compress = dict_data.get("compressed", [])

    size_data = 0
    size_compressed_data = 0

    for key in key_compress:
        _data = pickle.dumps(dict_data[key])
        size_data += len(_data)

        compressed_data = lzma.compress(_data)
        size_compressed_data += len(compressed_data)

        dict_data[key] = compressed_data

    return size_compressed_data, size_data


def dataproto_decompress(dict_data: dict[str, DataProto]) -> dict[str, DataProto]:
    key_compress = dict_data.get("compressed", [])

    for key in key_compress:
        compressed_data = lzma.decompress(dict_data[key])
        _data = pickle.loads(compressed_data)
        dict_data[key] = _data

    dict_data["compressed"] = []
