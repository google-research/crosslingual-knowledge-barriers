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


model="cross-ling-know/llama3-8b-wiki103-mixed-lang-sentence8words"
# model="cross-ling-know/llama3-8b-wiki103-mixed-lang-sentence"
# model="cross-ling-know/llama3-8b-wiki103-en"
arch="meta-llama/Meta-Llama-3-8B"


# crosslingual mixup MMLU
trans_lan="mixup"
python eval.py --model ${model} --arch ${arch} --ntrain 0 --config ${trans_lan} --test_trans_lan ${trans_lan}

# English MMLU
python eval.py --model ${model} --arch ${arch} --ntrain 0 --config en --test_trans_lan ""

# crosslingual MMLU

for pre in "question" "options" "full"  "gt" "gt_question"  "onewrong" "threewrong" 
do
for lan in  "de" "fr" "es" "it"
do
trans_lan=${pre}_${lan}
echo $trans_lan
python eval.py --model ${model} --arch ${arch} --ntrain 0 --config ${trans_lan} --test_trans_lan ${trans_lan}
done
done