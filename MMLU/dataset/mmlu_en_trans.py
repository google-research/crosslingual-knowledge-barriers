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

import os
import pandas as pd
from .mmlu import MMLUProcessor, INTERV_PROMPTS


class MMLUEnTransInstructProcessor(MMLUProcessor):
    def __init__(self, choices=None, interv_prompt_id=-1, data_dir=None):
        super().__init__(choices=choices, interv_prompt_id=interv_prompt_id)
        self.data_dir = data_dir

    def format_example(self, df, idx, include_answer=True, dev_df_en=None):
        k = df.shape[1] - 2  # number of choices
        prompt = "Question: " + df.iloc[idx, 0]

        for j in range(k):
            prompt += "\n{}. {}".format(self.choices[j], df.iloc[idx, j + 1])

        if include_answer:

            prompt += "\nAnswer: \nTranslate the question and options into English, and then answer.\n" + "Question: "

            # English version question
            prompt += dev_df_en.iloc[idx, 0]
            # English version options
            for j in range(k):
                prompt += "\n{}. {}".format(self.choices[j],
                                            dev_df_en.iloc[idx, j + 1])

            prompt += "\nAnswer:"
            # add demonstration answer
            dataset_answer = df.iloc[idx, k + 1]  # e.g., "A"
            # get the index, e.g., "A" has the index 0
            option_idex = self.dataset_choices.index(dataset_answer)
            # map it into the choice ID given the same index
            answer = self.choices[option_idex]
            prompt += " {}\n\n".format(answer)
        else:
            prompt += "\nAnswer: \nTranslate the question and options into English, and then answer.\n" + "Question: "

        return prompt

    def gen_prompt(self, train_df, subject, k=-1, dev_df_en=None):

        instruction = "The following are multiple choice questions (with answers) about {}.".format(
            self.format_subject(subject))

        # add intervention prompt
        if self.interv_prompt_id in INTERV_PROMPTS.keys():
            prompt = instruction + \
                " {}\n\n".format(INTERV_PROMPTS[self.interv_prompt_id])
        else:
            prompt = instruction + "\n\n"

        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            # few-shot demonstration
            prompt += self.format_example(train_df, i,
                                          include_answer=True, dev_df_en=dev_df_en)

        return prompt

    def gen_test_prompt(self, ntrain, test_df, dev_df, idx, subject):
        if subject not in self.subject_train_prompt.keys():
            # load english prompt
            dev_df_en = pd.read_csv(os.path.join(
                self.data_dir, "dev", subject + "_dev.csv"), header=None)[:ntrain]
            train_prompt = self.gen_prompt(
                dev_df, subject, ntrain, dev_df_en=dev_df_en)
            self.subject_train_prompt[subject] = train_prompt
        else:
            train_prompt = self.subject_train_prompt[subject]

        prompt_end = self.format_example(
            test_df, idx, include_answer=False, dev_df_en=None)
        prompt = train_prompt + prompt_end
        label = test_df.iloc[idx, test_df.shape[1] - 1]

        return prompt, label
