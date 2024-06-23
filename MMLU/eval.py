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


import argparse
import json
import os
import sys
from utils import load_llm

# setup python path so that we can import from third_party
third_party_path = os.path.join(os.path.abspath('..'), 'third_party')
sys.path.append(third_party_path)
from hendrycks_mmlu.lib_mmlu import eval_subjects, eval_subjects_hf_dataset

def main(args):

    prefix = os.path.join(os.path.join(
        args.save_dir, "{}_{}shot".format(args.model, args.ntrain)))
    if args.interv_prompt_id != -1:
        prefix += f"_interv{args.interv_prompt_id}"

    if (args.infer_mode == "generation") and ("openai" not in args.model):
        args.config += "_token"

    print("args.config", args.config, "prefix", prefix)
    output_folder = os.path.join(prefix,  args.config)

    results_file = os.path.join(prefix,  f"result_{args.config}.json")
    if os.path.exists(results_file):
        print("exists", results_file)
        return

    os.makedirs(output_folder, exist_ok=True)

    args.choices = ["A", "B", "C", "D"]
    # args.choices = ["1", "2", "3", "4"]
    # args.choices = ["C", "D", "E", "F"]

    llm = load_llm(args)

    # instruction: frist translate it into english, then answer
    if args.interv_prompt_id == 2:
        from dataset.mmlu_en_trans import MMLUEnTransInstructProcessor
        mmlu_proc = MMLUEnTransInstructProcessor(
            choices=args.choices, interv_prompt_id=args.interv_prompt_id,  data_dir=args.data_dir)

    else:
        from dataset.mmlu import MMLUProcessor
        mmlu_proc = MMLUProcessor(
            choices=args.choices, interv_prompt_id=args.interv_prompt_id, )

    # read translated dev data as demonstration
    dev_mode = "dev" if len(args.dev_trans_lan) == 0 else f"dev_{args.dev_trans_lan}"
    dev_folder=os.path.join(args.data_dir, dev_mode)
    # read translated test data as test group
    test_mode = "test" if len(args.test_trans_lan) == 0 else f"test_{args.test_trans_lan}"
    test_folder=os.path.join(args.data_dir, test_mode)

    from dataset.mmlu import subcategories
    # read all subjects
    subjects = sorted(list(subcategories.keys()))

    if (args.test_trans_lan=="mixup") and (os.path.exists(test_folder)==False):
        # use huggingface dataset for mixup MMLU
        print("load from hugginface dataset: cross-ling-know/mixup-lang-mmlu")
        results = eval_subjects_hf_dataset(subjects=subjects,
                                hf_dataset_name="cross-ling-know/mixup-lang-mmlu",
                                llm=llm,
                                mmlu_proc=mmlu_proc,
                                output_folder=output_folder,
                                ntrain=args.ntrain,
                                choices=args.choices,
                                infer_mode=args.infer_mode)

    else:
        # use local csv files as dataset
        results = eval_subjects(subjects=subjects,
                                dev_folder=dev_folder,
                                test_folder=test_folder,
                                llm=llm,
                                mmlu_proc=mmlu_proc,
                                output_folder=output_folder,
                                ntrain=args.ntrain,
                                choices=args.choices,
                                infer_mode=args.infer_mode)
    

    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5,
                        help="Number of few-show demonstrations to use (default: 5) from dev dataset")
    parser.add_argument("--data_dir", "-d", type=str,
                        default="data", help="Directory where the data is stored")
    parser.add_argument("--save_dir", "-s", type=str, default="results",
                        help="Directory where the results will be saved")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name or path")
    parser.add_argument("--arch", type=str, default=None,
                        help="Model architecture")
    parser.add_argument("--config", "-c", type=str, default="en",
                        help="Configuration name used when saving the results")

    parser.add_argument("--infer_mode", type=str, default="logits", choices=["logits", "generation"],
                        help="Inference mode for next token, either token with maximal 'logits' or free-form next token 'generation'")
    parser.add_argument("--max_token", type=int, default=32,
                        help="Maximum number of tokens if using `generation` infer_mode")

    parser.add_argument('--test_trans_lan', type=str,
                        default='', help="Language to translate test data to; default is English")
    parser.add_argument('--dev_trans_lan', type=str, default='',
                        help="Language to translate dev data tto; default is English")
    parser.add_argument('--interv_prompt_id', type=int, default=-1,
                        help="Intervention prompt ID (default: -1, meaning no prompt)")
    args = parser.parse_args()

    main(args)
