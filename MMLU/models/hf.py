# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from .base import LLMBase


class CasualLM(LLMBase):
    """Huggingface Casual Language Models.

    Parameters:
    - model_path (str): The path/name for the desired langauge model.
    """

    def __init__(self, model_path=None, arch=None, max_tokens=1024, infer_mode="generation"):

        if arch is None:
            self.arch = model_path
        else:
            self.arch = arch
        
        self.tokenizer_use_fast = True
        self.max_tokens = max_tokens
        self.verbose = True
        self.infer_mode = infer_mode

        super().__init__(model_path=model_path)

    def load_model(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            device_map="auto",
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(self.arch)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        self.model = model
        self.tokenizer = tokenizer

        print(
            f"> Loading the provided {self.arch} checkpoint from '{model_path}'.")

    def query(self, prompt, choices=None):
        """
        Query an open-source model with a given text prompt.

        Parameters:
        - prompt (str): The text prompt to query the model.

        Returns[]:
        - str: The model's output.
        - list: Predicted logits for options
        """
        # print(prompt)
        if self.infer_mode == "generation":
            return self.query_generation(prompt)
        elif self.infer_mode == "logits":
            return self.query_logits(prompt, choices)

    @torch.no_grad()
    def query_generation(self, prompt):
        try:
            model_inputs = self.tokenizer(
                prompt, return_tensors="pt").to(self.model.device)
            generated_ids = self.model.generate(
                **model_inputs, max_new_tokens=self.max_tokens, do_sample=True)
            pred = self.tokenizer.batch_decode(generated_ids[:, model_inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True)[0]
        except Exception as e:
            print(e)
            pred = ""
        return pred, None

    @torch.no_grad()
    def query_logits(self, prompt, choices):
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.to(self.model.device)
        logits = self.model(input_ids=input_ids).logits[0, -1]
        target_logits = []
        for choice in choices:
            target_logits.append(logits[self.tokenizer(choice).input_ids[-1]])
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    target_logits
                ).float(),
                dim=0,
            )
            .detach()
            .cpu()
            .numpy()
        )
        choices_dict = {}
        for idx in range(len(choices)):
            choices_dict[idx] = choices[idx]

        pred = choices_dict[np.argmax(probs)]

        return pred, probs


if __name__ == '__main__':
    # Testing purposes
    model = CasualLM('gpt2')
    print(model.query(['hello. how are you?', 'what is your name?']))
    print("DONE")
