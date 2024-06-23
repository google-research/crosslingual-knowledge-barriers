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

import time
from copy import deepcopy

from .base import LLMBase
import openai


class ChatGPT(LLMBase):
    def __init__(
        self,
        model_path=None,
        max_attempts=100,
        max_tokens=2048,
        temperature=0,
    ):

        openai.api_key = ""  # replace it with your key
        openai.api_base = ""

        self.max_attempts = max_attempts
        self.delay_seconds = 1
        self.model = model_path.replace("openai/", "")
        self.parameters = {"max_tokens": max_tokens,
                           "temperature": temperature}
        

    def query(self, prompt, choices=None):
        pred = self.chat_query(prompt)
        logits = None
        return pred, logits

    def chat_query(self, prompt, messages=None):

        n_attempt = 0
        params = deepcopy(self.parameters)

        if messages is None:
            messages = [{"role": "user", "content": prompt}]

        print("messages", messages)
        while n_attempt < self.max_attempts:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model, messages=messages, **params
                )
                response = completion["choices"][0]["message"]["content"]
                # print(completion)
                return response
            except Exception as e:
                # Catch any exception that might occur and print an error message
                print(f"An error occurred: {e}, retry {n_attempt}")
                n_attempt += 1
                time.sleep(self.delay_seconds*n_attempt)

        if n_attempt == self.max_attempts:
            print("Max number of attempts reached")
            return ""

        return ""

