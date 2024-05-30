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

class LLMBase:
    def __init__(self, model_path=None, api_key=None):
        """
        Initialize a Large Language Model (LLM).

        Parameters:

        - model_path (str): The file path or URL to the model. Default is None.
        - api_key (str): The API key for querying closed-source models. Default is None.

        """

        self.model_path = model_path  # file path or URL that points to the model
        self.api_key = api_key  # API key for accessing LLMs (e.g., ChatGPT)
        self.load_model()

    def load_model(self):
        pass

    def query(self, text):
        """
        Query a model with a given text prompt.

        Parameters:
        - text (str): The text prompt to query the model.

        Returns:
        - str: The model's output.
        """
        pass
