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

import pandas as pd
import os
from googletrans import Translator

translator = Translator()
task = "astronomy"
trans_lan = "fr"
choices = ["A", "B", "C", "D"]
import random

random.seed(42)
split = "test"
# split="dev"
TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


def translate_test_task_full(task, trans_lan):
    # Step 1: Read the CSV file into a DataFrame
    save_folder = os.path.join("data", f"{split}_full_{trans_lan}")
    os.makedirs(save_folder, exist_ok=True)
    save_fname = os.path.join(save_folder, task + f"_{split}.csv")
    if os.path.isfile(save_fname):
        return  ## already have!

    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    for idx in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = test_df.shape[1] - 2  # number of choices

        # batch_english=[]
        prompt = test_df.iloc[idx, 0]
        try:
            test_df.iloc[idx, 0] = translator.translate(
                prompt, src="en", dest=trans_lan
            ).text
        except:
            print("translation error for prompt: ", prompt)

        for j in range(k):
            try:
                option_english = test_df.iloc[idx, j + 1]
                test_df.iloc[idx, j + 1] = translator.translate(
                    option_english, src="en", dest=trans_lan
                ).text
            except:
                print("translation error for option: ", option_english)

        print(trans_lan, test_df.iloc[idx, 0])
        print("****")

    test_df.to_csv(save_fname, header=None, index=False)


def translate_test_task_gt_translate(task, trans_lan):
    # Step 1: Read the CSV file into a DataFrame
    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    for idx in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = test_df.shape[1] - 2  # number of choices

        # prompt_end = format_example(test_df, i, include_answer=False)
        answer_choice = test_df.iloc[idx, k + 1]
        answer_index = choices.index(answer_choice)

        option_english = test_df.iloc[idx, answer_index + 1]
        print("en", option_english)
        try:
            option_trans = translator.translate(
                option_english, src="en", dest=trans_lan
            ).text
        except:
            option_trans = option_english

        print(trans_lan, option_trans)
        print("****")
        test_df.iloc[idx, answer_index + 1] = option_trans

    # Step 3: Write the modified DataFrame back to a new CSV file
    save_folder = os.path.join("data", f"{split}_gt_{trans_lan}")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


def translate_test_task_mixup_translate(task):
    # Step 1: Read the CSV file into a DataFrame

    trans_lan_full = ["en", "fr", "de", "es", "it"]
    random.shuffle(trans_lan_full)

    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    for idx in range(test_df.shape[0]):
        # get prompt and make sure it fits
        k = test_df.shape[1] - 2  # number of choices

        prompt = test_df.iloc[idx, 0]
        try:
            trans_lan = trans_lan_full[0]
            if trans_lan != "en":
                test_df.iloc[idx, 0] = translator.translate(
                    prompt, src="en", dest=trans_lan
                ).text
        except:
            print("translation error for prompt: ", prompt)

        for j in range(k):
            trans_lan = trans_lan_full[j + 1]
            try:
                if trans_lan != "en":
                    option_english = test_df.iloc[idx, j + 1]
                    test_df.iloc[idx, j + 1] = translator.translate(
                        option_english, src="en", dest=trans_lan
                    ).text
            except:
                print("translation error for option: ", option_english)

        print(idx, trans_lan_full[0], test_df.iloc[idx, 0])
        print("****")

    # Step 3: Write the modified DataFrame back to a new CSV file
    save_folder = os.path.join("data", f"{split}_mixup")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


def translate_test_task_mixup(task):
    # Step 1: Read the CSV file into a DataFrame

    trans_lan_full = ["en", "fr", "de", "es", "it"]
    trans_test_df_dict = {}
    for lan in trans_lan_full[1:]:
        trans_test_df_dict[lan] = pd.read_csv(
            os.path.join("data", f"{split}_full_{lan}", task + f"_{split}.csv"),
            header=None,
        )

    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )

    for idx in range(test_df.shape[0]):
        random.shuffle(trans_lan_full)
        # get prompt and make sure it fits
        k = test_df.shape[1] - 2  # number of choices

        prompt = test_df.iloc[idx, 0]
        try:
            trans_lan = trans_lan_full[0]
            if trans_lan != "en":
                # test_df.iloc[idx, 0]= translator.translate(prompt, src='en', dest=trans_lan).text
                test_df.iloc[idx, 0] = trans_test_df_dict[trans_lan].iloc[idx, 0]

        except:
            print("translation error for prompt: ", prompt)

        for j in range(k):
            trans_lan = trans_lan_full[j + 1]
            try:
                if trans_lan != "en":
                    option_english = test_df.iloc[idx, j + 1]
                    # test_df.iloc[idx, j+1]= translator.translate(option_english, src='en', dest=trans_lan).text
                    test_df.iloc[idx, j + 1] = trans_test_df_dict[trans_lan].iloc[
                        idx, j + 1
                    ]
            except:
                print("translation error for option: ", option_english)

        print(idx, trans_lan_full[0], test_df.iloc[idx, 0])
        print("****")

    # Step 3: Write the modified DataFrame back to a new CSV file
    save_folder = os.path.join("data", f"{split}_mixup")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


def translate_test_task_wrong_option(task, trans_lan, onewrong=True):
    # Step 1: Read the CSV file into a DataFrame
    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    # trans_lan_full= ['fr','de','es','it']
    # trans_test_df={}

    # for lan in trans_lan_full:

    trans_test_df = pd.read_csv(
        os.path.join("data", f"{split}_full_{trans_lan}", task + f"_{split}.csv"),
        header=None,
    )

    for idx in range(test_df.shape[0]):
        k = test_df.shape[1] - 2  # number of choices

        answer_choice = test_df.iloc[idx, k + 1]
        answer_index = choices.index(answer_choice)
        wrong_options_index = [j for j in range(1, k) if j != answer_index]

        if onewrong:
            wrong_option_idx = random.choice(wrong_options_index)
            try:
                test_df.iloc[idx, wrong_option_idx + 1] = trans_test_df.iloc[
                    idx, wrong_option_idx + 1
                ]
            except Exception as e:
                print(e)
        else:
            for wrong_option_idx in wrong_options_index:
                try:
                    test_df.iloc[idx, wrong_option_idx + 1] = trans_test_df.iloc[
                        idx, wrong_option_idx + 1
                    ]
                except Exception as e:
                    print(e)
        print(trans_lan, test_df.iloc[idx, wrong_option_idx + 1])
        print("****")

    # Step 3: Write the modified DataFrame back to a new CSV file
    if onewrong:
        save_folder = os.path.join("data", f"{split}_onewrong_{trans_lan}")
    else:
        save_folder = os.path.join("data", f"{split}_threewrong_{trans_lan}")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


def translate_test_task_gt(task, trans_lan):

    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    trans_test_df = pd.read_csv(
        os.path.join("data", f"{split}_full_{trans_lan}", task + f"_{split}.csv"),
        header=None,
    )

    for idx in range(test_df.shape[0]):
        k = test_df.shape[1] - 2  # number of choices

        answer_choice = test_df.iloc[idx, k + 1]
        answer_index = choices.index(answer_choice)
        try:
            test_df.iloc[idx, answer_index + 1] = trans_test_df.iloc[
                idx, answer_index + 1
            ]

        except Exception as e:
            print(e)

        print(trans_lan, test_df.iloc[idx, 0], test_df.iloc[idx, answer_index + 1])
        print("****")

    # Step 3: Write the modified DataFrame back to a new CSV file
    save_folder = os.path.join("data", f"{split}_gt_{trans_lan}")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


def translate_test_task_gt_question(task, trans_lan):

    test_df = pd.read_csv(
        os.path.join("data", split, task + f"_{split}.csv"), header=None
    )
    trans_test_df = pd.read_csv(
        os.path.join("data", f"{split}_full_{trans_lan}", task + f"_{split}.csv"),
        header=None,
    )

    for idx in range(test_df.shape[0]):
        k = test_df.shape[1] - 2  # number of choices

        answer_choice = test_df.iloc[idx, k + 1]
        answer_index = choices.index(answer_choice)
        try:
            test_df.iloc[idx, answer_index + 1] = trans_test_df.iloc[
                idx, answer_index + 1
            ]
            test_df.iloc[idx, 0] = trans_test_df.iloc[idx, 0]

        except Exception as e:
            print(e)

        print(trans_lan, test_df.iloc[idx, 0], test_df.iloc[idx, answer_index + 1])
        print("****")

    # Step 3: Write the modified DataFrame back to a new CSV file
    save_folder = os.path.join("data", f"{split}_gt_question_{trans_lan}")
    os.makedirs(save_folder, exist_ok=True)
    test_df.to_csv(
        os.path.join(save_folder, task + f"_{split}.csv"), header=None, index=False
    )


# FR- French, DE- German, ES – Spanish, IT- Italian


for task in TASKS:

    translate_test_task_mixup(task)

    # for trans_lan in ['fr','de','es','it']:
    #     # translate_test_task_full(task,trans_lan) # no

    #     translate_test_task_gt_question(task,trans_lan)
    #     translate_test_task_gt(task,trans_lan)
    #     translate_test_task_wrong_option(task,trans_lan, onewrong=True)
    #     translate_test_task_wrong_option(task,trans_lan, onewrong=False)
    #     print("done!",task, trans_lan )
