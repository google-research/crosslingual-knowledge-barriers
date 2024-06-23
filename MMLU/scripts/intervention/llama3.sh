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

model="meta-llama/Meta-Llama-3-8B"
arch="meta-llama/Meta-Llama-3-8B"

trans_lan=mixup

# zero shot with Multilingual-Aware instruction 0
python eval.py --model ${model} --arch ${arch} --ntrain 0 --config ${trans_lan} --test_trans_lan ${trans_lan} --interv_prompt_id 0

# zero shot with Multilingual-Aware instruction 1
python eval.py --model ${model} --arch ${arch} --ntrain 0 --config ${trans_lan} --test_trans_lan ${trans_lan} --interv_prompt_id 1


# five shot demonstrations for Translate-Then-Answer
max_token=256
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config ${trans_lan}_${max_token} --test_trans_lan ${trans_lan} --dev_trans_lan ${trans_lan} --interv_prompt_id 2 --infer_mode "generation" --max_token ${max_token}
