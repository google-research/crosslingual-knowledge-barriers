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

INTERV_PROMPTS = {
    0: "Keep in mind that the question and options may be presented in various languages.",
    1: "Remember that the question and options can be in different languages.",
}


class MMLUProcessor:
    def __init__(self):
        self.dataset_choices = ["A", "B", "C", "D"]

    def format_subject(self, subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry

        s = s.replace(" perturb", "")  # remove irrelvant word in subject title
        return s

    def format_example(self, df, idx, include_answer=True, choices=None):

        if choices is None:
            choices = self.dataset_choices

        prompt = df.iloc[idx, 0]
        k = df.shape[1] - 2

        # if we want to have six options:
        # prompt += "\n{}. {}".format("A", "Happiness")
        # prompt += "\n{}. {}".format("B", "A herd of cats")

        for j in range(k):
            prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
        prompt += "\nAnswer:"

        if include_answer:
            dataset_answer = df.iloc[idx, k + 1]  # for example "A"
            option_idex = self.dataset_choices.index(
                dataset_answer
            )  # get the index, for example "0"
            answer = choices[
                option_idex
            ]  # map it into the new choice given the same index
            prompt += " {}\n\n".format(answer)
        return prompt

    def gen_prompt(self, train_df, subject, k=-1, interv_prompt_id=-1, choices=None):

        if interv_prompt_id in INTERV_PROMPTS.keys():
            prompt = "The following are multiple choice questions (with answers) about {}. {}\n\n".format(
                self.format_subject(subject), INTERV_PROMPTS[interv_prompt_id]
            )
        else:
            prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                self.format_subject(subject)
            )

        if k == -1:
            k = train_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                train_df, i, include_answer=True, choices=choices
            )  # few-shot demonstration
        return prompt

    def gen_test_prompt(
        self, ntrain, test_df, dev_df, idx, subject, interv_prompt_id=-1, choices=None
    ):

        train_prompt = self.gen_prompt(
            dev_df, subject, ntrain, interv_prompt_id=interv_prompt_id, choices=choices
        )
        prompt_end = self.format_example(
            test_df, idx, include_answer=False, choices=choices
        )
        prompt = train_prompt + prompt_end

        label = test_df.iloc[idx, test_df.shape[1] - 1]

        return prompt, label


subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

categories = {
    "STEM": [
        "physics",
        "chemistry",
        "biology",
        "computer science",
        "math",
        "engineering",
    ],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}
