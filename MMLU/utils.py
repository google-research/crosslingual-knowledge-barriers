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

def load_llm(args):
    if "openai" in args.model:
        from models.openai import ChatGPT
        llm = ChatGPT(model_path=args.model, max_tokens=args.max_token)
    else:
        from models.hf import CasualLM
        llm = CasualLM(model_path=args.model, arch=args.arch,
                       max_tokens=args.max_token, infer_mode=args.infer_mode)
    return llm

