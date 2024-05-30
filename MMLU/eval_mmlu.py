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
import numpy as np
import pandas as pd

from datasets.mmlu import categories, subcategories, MMLUProcessor


def load_llm(args):
    if "openai" in args.model:
        from models.openai import ChatGPT

        llm = ChatGPT(model_path=args.model, max_tokens=args.max_token)
    else:
        from models.hf import CasualLM

        llm = CasualLM(
            model_path=args.model, max_tokens=args.max_token, infer_mode=args.infer_mode
        )
    return llm


def eval(args, subject, llm, dev_df, test_df, choices):
    cors = []
    all_probs = []
    answers = choices[: test_df.shape[1] - 2]
    all_prompts = []
    all_predicts = []
    mmlu_proc = MMLUProcessor()

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        prompt, label = mmlu_proc.gen_test_prompt(
            args.ntrain, test_df, dev_df, i, subject, args.interv_prompt_id, choices
        )

        pred, probs = llm.query(prompt, choices)

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        all_predicts.append(pred)
        all_prompts.append(prompt)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


def main(args):
    if args.interv_prompt_id != -1:
        prefix = os.path.join(
            os.path.join(
                args.save_dir,
                "{}_{}shot_interv{}".format(
                    args.model, args.ntrain, args.interv_prompt_id
                ),
            )
        )
    else:
        prefix = os.path.join(
            os.path.join(args.save_dir, "{}_{}shot".format(args.model, args.ntrain))
        )

    print("args.config", args.config, "prefix", prefix)

    results_file = os.path.join(prefix, f"result_{args.config}.json")

    if os.path.exists(os.path.join(prefix, args.config)):
        print("exit", os.path.join(prefix, args.config))

    choices = ["A", "B", "C", "D"]
    # choices = ["1", "2", "3", "4"]
    # choices = ["C", "D", "E", "F"]

    llm = load_llm(args)

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    os.makedirs(os.path.join(prefix, args.config), exist_ok=True)
    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}
    results = {"subject": {}, "subcategories": {}, "categories": {}}

    for subject in subjects:
        dev_folder = (
            "dev" if len(args.dev_trans_lan) == 0 else f"dev_{args.dev_trans_lan}"
        )
        test_folder = (
            "test" if len(args.test_trans_lan) == 0 else f"test_{args.test_trans_lan}"
        )
        if args.ntrain > 0:
            dev_df = pd.read_csv(
                os.path.join(args.data_dir, dev_folder, subject + "_dev.csv"),
                header=None,
            )[: args.ntrain]
        else:
            dev_df = None
        test_df = pd.read_csv(
            os.path.join(args.data_dir, test_folder, subject + "_test.csv"), header=None
        )

        cors, acc, probs = eval(args, subject, llm, dev_df, test_df, choices=choices)
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        test_df["{}_correct".format(args.model)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(args.model, choice)] = probs[:, j]
        test_df.to_csv(
            os.path.join(prefix, args.config, "{}.csv".format(subject)),
            index=None,
        )
        results["subject"][subject] = acc

    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        results["subcategories"][subcat] = subcat_acc
        print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        results["categories"][cat] = cat_acc
        print("Average accuracy {:.3f} - {}".format(cat_acc, cat))
    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    with open(results_file, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--config", "-c", type=str, default="en_anw_blank")

    parser.add_argument(
        "--infer_mode", type=str, default="logits", choices=["logits", "generation"]
    )
    parser.add_argument("--max_token", type=int, default=32)
    parser.add_argument("--cot", action="store_true")

    parser.add_argument("--test_trans_lan", type=str, default="")
    parser.add_argument("--dev_trans_lan", type=str, default="")
    parser.add_argument("--interv_prompt_id", type=int, default=-1)  # no prompt
    args = parser.parse_args()

    main(args)
