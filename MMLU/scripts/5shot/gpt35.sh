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



model="openai/gpt-3.5-turbo-0125"
# model="openai/gpt-4-0613"


# 5-shot

# crosslingual MMLU
trans_lan="mixup"
# use mixup (same biased) demo
python eval.py --model ${model} --ntrain 5 --config ${trans_lan} --test_trans_lan ${trans_lan} --dev_trans_lan ${trans_lan} --max_token 1 --infer_mode "generation"
# use english demo
python eval.py --model ${model} --ntrain 5 --config ${trans_lan}_en_demo --test_trans_lan ${trans_lan} --dev_trans_lan "" --max_token 1 --infer_mode "generation"




# English MMLU
python eval.py --model ${model}  --ntrain 5 --config en --test_trans_lan "" --max_token 1 --infer_mode "generation"




