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

for split in dev test
do
    # first, do full translation
    python translate.py --split $split --languages fr de es it --mode full

    # then, create crosslingual dataset
    for mode in mixup question options gt_question gt wrong_option
        do
        python translate.py --split $split --languages fr de es it --mode $mode
    done
done