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
import os

import numpy as np
import pandas as pd
import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


instructions = {
    "de": "Im Folgenden finden Sie Multiple-Choice-Fragen (mit Antworten) zu Harry Potter.",
    "en": "The following are multiple choice questions (with answers) about Harry Potter.",
    "es": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre Harry Potter.",
    "fr": "Vous trouverez ci-dessous des questions à choix multiples (avec réponses) sur Harry Potter.",
    "it": "Di seguito sono riportate domande a scelta multipla (con risposte) su Harry Potter.",
}

langcode = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
}


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model.seqlen = model.config.max_position_embeddings
    return model, tokenizer


def prepare_prompts(record, instruction):

    question = record["question"]
    choice_A = record["A"]
    choice_B = record["B"]
    choice_C = record["C"]
    choice_D = record["D"]
    prompt = (
        instruction
        + "\n\n"
        + f"{question}\nA. {choice_A}\nB. {choice_B}\nC. {choice_C}\nD. {choice_D}\nAnswer: "
    )

    return prompt


def query_logits(model, tokenizer, prompt, choices=["A", "B", "C", "D"]):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    logits = model(input_ids=input_ids).logits[0, -1]
    target_logits = []
    for choice in choices:
        target_logits.append(logits[tokenizer(choice).input_ids[-1]])
    probs = (
        torch.nn.functional.softmax(
            torch.tensor(target_logits).float(),
            dim=0,
        )
        .detach()
        .cpu()
        .numpy()
    )
    choices_dict = {}
    for idx in range(len(choices)):
        choices_dict[idx] = choices[idx]

    pred = choices_dict[np.argmax(probs)]

    return pred, probs


def eval_multiple_choice(
    model,
    tokenizer,
    eval_data,
):

    res = []

    output_dir = os.path.join(args.save_dir, args.model_name)

    file_name = f"eval_{args.lang}"

    instruction = instructions[args.lang]

    os.makedirs(output_dir, exist_ok=True)

    for record in tqdm.tqdm(eval_data):

        prompt = prepare_prompts(record, instruction)
        pred, _ = query_logits(model, tokenizer, prompt, choices=["A", "B", "C", "D"])

        is_correct = pred == record["GT"]

        res.append(
            {
                "question": record["question"],
                "gt_ans": record["GT"],
                "prompt": prompt,
                "ans": pred,
                "correct": is_correct,
            }
        )

        df = pd.DataFrame(res)
        df.to_csv(os.path.join(output_dir, f"{file_name}.csv"))
    print(f'HP-Quiz Accuracy: {df["correct"].mean()}')


def main(args):
    model, tokenizer = load_model(args.model_name)

    data = load_dataset("cross-ling-know/HarryPotter-Quiz")
    eval_data = data[langcode[args.lang]].to_pandas().to_dict("records")

    eval_multiple_choice(model=model, tokenizer=tokenizer, eval_data=eval_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-ling-know/llama2-7b-wiki2-en",
        help="The local path or HuggingFace link of the evaluated model",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="out/",
        help="The folder to save evaluation results",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "fr", "de", "es", "it"],
        help="The code for the language to be evaluated",
    )

    args = parser.parse_args()
    main(args)
