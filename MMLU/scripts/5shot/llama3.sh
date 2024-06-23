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

# 5-shot


# crosslingual mixup MMLU
trans_lan="mixup"
# use mixup (same biased) demo
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config ${trans_lan} --test_trans_lan ${trans_lan} --dev_trans_lan ${trans_lan}
# use english demo
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config ${trans_lan}_en_demo --test_trans_lan ${trans_lan} --dev_trans_lan ""

# English MMLU
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config en --test_trans_lan "" --dev_trans_lan ""


# crosslingual MMLU
for pre in  "full"  "question" "options" "gt_question" "gt"   "onewrong" "threewrong" 
do
for lan in  "de" "fr" "es" "it"
do
trans_lan=${pre}_${lan}
echo $trans_lan
# use same biased demo
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config ${trans_lan} --test_trans_lan ${trans_lan} --dev_trans_lan ${trans_lan}
# use english demo
python eval.py --model ${model} --arch ${arch} --ntrain 5 --config ${trans_lan}_en_demo --test_trans_lan ${trans_lan} --dev_trans_lan ""
done
done
