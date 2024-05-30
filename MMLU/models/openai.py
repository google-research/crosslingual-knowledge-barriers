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
import tiktoken
from copy import deepcopy

from .base import LLMBase
import openai


class ChatGPT(LLMBase):
    def __init__(
        self,
        api_key=None,
        model_path=None,
        max_attempts=100,
        max_tokens=2048,
        temperature=0,
    ):

        openai.api_key = ""
        openai.api_base = ""

        self.max_attempts = max_attempts
        self.delay_seconds = 1
        self.model = model_path.replace("openai/", "")
        self.parameters = {"max_tokens": max_tokens, "temperature": temperature}

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
            # num_tokens = num_tokens_from_messages(messages)
            # if num_tokens > 4096 - params['max_tokens']:
            #     params['max_tokens'] = 4096 - num_tokens - 20
            #     if params['max_tokens'] < 1:
            #         return ''  # cannot generate anything.
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model, messages=messages, **params
                )
                response = completion["choices"][0]["message"]["content"]
                # print(completion)
                return response
            except Exception as e:
                # Catch any exception that might occur and print an error message
                print(f"An error occurred: {e}")
                n_attempt += 1
                time.sleep(self.delay_seconds)

        if n_attempt == self.max_attempts:
            print("Max number of attempts reached")
            return ""

        return ""


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
