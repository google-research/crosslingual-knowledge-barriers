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

import random
import pandas as pd
import os
import argparse
from googletrans import Translator
translator = Translator()
choices = ["A", "B", "C", "D"]
random.seed(42)


TASKS = [
    'abstract_algebra',
    'anatomy',
    'astronomy',
    'business_ethics',
    'clinical_knowledge',
    'college_biology',
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_medicine',
    'college_physics',
    'computer_security',
    'conceptual_physics',
    'econometrics',
    'electrical_engineering',
    'elementary_mathematics',
    'formal_logic',
    'global_facts',
    'high_school_biology',
    'high_school_chemistry',
    'high_school_computer_science',
    'high_school_european_history',
    'high_school_geography',
    'high_school_government_and_politics',
    'high_school_macroeconomics',
    'high_school_mathematics',
    'high_school_microeconomics',
    'high_school_physics',
    'high_school_psychology',
    'high_school_statistics',
    'high_school_us_history',
    'high_school_world_history',
    'human_aging',
    'human_sexuality',
    'international_law',
    'jurisprudence',
    'logical_fallacies',
    'machine_learning',
    'management',
    'marketing',
    'medical_genetics',
    'miscellaneous',
    'moral_disputes',
    'moral_scenarios',
    'nutrition',
    'philosophy',
    'prehistory',
    'professional_accounting',
    'professional_law',
    'professional_medicine',
    'professional_psychology',
    'public_relations',
    'security_studies',
    'sociology',
    'us_foreign_policy',
    'virology',
    'world_religions']


def save_translated_df(df, folder, filename):
    os.makedirs(folder, exist_ok=True)
    df.to_csv(os.path.join(folder, filename), header=None, index=False)


def translate_text(text, src='en', dest='fr'):
    try:
        return translator.translate(text, src=src, dest=dest).text
    except Exception as e:
        print(f"Translation error for text: {text}\n{e}")
        return text


def translate_test_task_full(split, task, trans_lan):
    save_folder = os.path.join('data', f"{split}_full_{trans_lan}")
    save_fname = os.path.join(save_folder, f"{task}_{split}.csv")
    if os.path.isfile(save_fname):
        return

    test_df = pd.read_csv(os.path.join(
        'data', split, f"{task}_{split}.csv"), header=None)
    for idx in range(test_df.shape[0]):
        # translate question
        test_df.iloc[idx, 0] = translate_text(
            test_df.iloc[idx, 0], dest=trans_lan)
        # translate options
        for j in range(test_df.shape[1] - 2):
            test_df.iloc[idx, j +
                         1] = translate_text(test_df.iloc[idx, j + 1], dest=trans_lan)

    save_translated_df(test_df, save_folder, f"{task}_{split}.csv")


def translate_test_task_mixup(split, task, languages):
    # languages: a list of translation languages
    trans_lan_full = ['en'] + languages
    print(trans_lan_full)
    trans_test_df_dict = {
        lan: pd.read_csv(os.path.join(
            'data', f"{split}_full_{lan}", f"{task}_{split}.csv"), header=None)
        for lan in trans_lan_full[1:]
    }
    test_df = pd.read_csv(os.path.join(
        'data', split, f"{task}_{split}.csv"), header=None)

    for idx in range(test_df.shape[0]):
        # random order of languages for question and options
        random.shuffle(trans_lan_full)

        # question
        trans_lan = trans_lan_full[0]
        if trans_lan != 'en':
            test_df.iloc[idx, 0] = trans_test_df_dict[trans_lan].iloc[idx, 0]

        # options
        k = test_df.shape[1] - 2  # number of choices
        for j in range(k):
            trans_lan = trans_lan_full[j + 1]
            if trans_lan != 'en':
                test_df.iloc[idx, j +
                             1] = trans_test_df_dict[trans_lan].iloc[idx, j + 1]

    save_folder = os.path.join('data', f"{split}_mixup")
    save_translated_df(test_df, save_folder, f"{task}_{split}.csv")


def translate_test_task_options(split, task, trans_lan, mode='options', onewrong=True):
    test_df = pd.read_csv(os.path.join(
        'data', split, f"{task}_{split}.csv"), header=None)
    trans_test_df = pd.read_csv(os.path.join(
        'data', f"{split}_full_{trans_lan}", f"{task}_{split}.csv"), header=None)
    k = test_df.shape[1] - 2  # number of options

    for idx in range(test_df.shape[0]):

        # trans question
        if mode in ['question', 'gt_question']:
            test_df.iloc[idx, 0] = trans_test_df.iloc[idx, 0]

        if mode in ['gt', 'gt_question']:
            # trans gt option
            answer_choice = test_df.iloc[idx, k + 1]
            answer_index = choices.index(answer_choice)
            test_df.iloc[idx, answer_index +
                         1] = trans_test_df.iloc[idx, answer_index+1]

        elif mode == 'options':
            # trans all options
            for j in range(k):
                test_df.iloc[idx, j + 1] = trans_test_df.iloc[idx, j + 1]
        elif mode == 'wrong_option':
            answer_choice = test_df.iloc[idx, k + 1]
            answer_index = choices.index(answer_choice)
            wrong_options_index = [j for j in range(k) if j != answer_index]
            if onewrong:
                wrong_option_idx = random.choice(wrong_options_index)
                test_df.iloc[idx, wrong_option_idx +
                             1] = trans_test_df.iloc[idx, wrong_option_idx + 1]
            else:
                for wrong_option_idx in wrong_options_index:
                    test_df.iloc[idx, wrong_option_idx +
                                 1] = trans_test_df.iloc[idx, wrong_option_idx + 1]
    save_mode = mode
    if mode == 'wrong_option':
        save_mode = "onewrong" if onewrong == True else "threewrong"
    save_folder = os.path.join('data', f"{split}_{save_mode}_{trans_lan}")
    save_translated_df(test_df, save_folder, f"{task}_{split}.csv")


def process_all_tasks(split, languages, mode):
    for task in TASKS:
        if mode == 'full':
            for trans_lan in languages:
                translate_test_task_full(split, task, trans_lan)
                print("done full!", task, trans_lan)
        elif mode == 'mixup':
            translate_test_task_mixup(split, task, languages)
            print("done mixup!", task)
        else:
            for trans_lan in languages:
                if mode in ['question', 'options', 'gt_question', 'gt']:
                    translate_test_task_options(
                        split, task, trans_lan, mode=mode)
                elif mode == "wrong_option":
                    for onewrong in [True, False]:
                        translate_test_task_options(
                            split, task, trans_lan, mode='wrong_option', onewrong=onewrong)
                print(f"done!", task, trans_lan)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process translation tasks.")

    parser.add_argument('--split', type=str, default='dev',  choices=['dev', 'test'],
                        help="The dataset split to process")
    parser.add_argument('--languages', type=str, nargs='+',
                        default=['fr', 'de', 'es', 'it'], 
                        help="List of languages for translation")
    parser.add_argument('--mode', type=str, choices=['full', 'mixup', 'question', 'options',
                        'gt_question', 'gt', 'wrong_option'], required=True, 
                        help="Mode of translation")

    args = parser.parse_args()
    print(args)
    # Main execution
    process_all_tasks(args.split, args.languages, args.mode)
