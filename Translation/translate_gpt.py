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
from typing import List, Dict

from openai import OpenAI
from tqdm import tqdm

LANG_DICT: Dict[str, str] = {
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "en": "English",
}

MODEL_ENDPOINTS: Dict[str, str] = {
    "gpt-4": "gpt-4-0613",
    "gpt-3.5": "gpt-3.5-turbo-0125",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate script using GPT models")
    parser.add_argument(
        "--model", type=str, default="gpt-4", choices=["gpt-4", "gpt-3.5"]
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/flores",
        help="Directory to load input data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gpt4",
        help="Directory to save translation results",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="en2x",
        choices=["x2en", "en2x"],
        help="Translation direction: 'x2en' for X to English, 'en2x' for English to X",
    )
    args = parser.parse_args()
    return args


def translate_text(
    model: str, client: OpenAI, text: str, source_lang: str, target_lang: str
) -> str:
    endpoint = MODEL_ENDPOINTS.get(model)
    if not endpoint:
        raise ValueError(f"Invalid model: {model}")

    response = client.chat.completions.create(
        model=endpoint,
        messages=[
            {
                "role": "system",
                "content": f"Translate the following {source_lang} text to {target_lang}:",
            },
            {"role": "user", "content": text},
        ],
        temperature=0.7,
        max_tokens=64,
        top_p=1,
    )
    return response.choices[0].message.content


def save_results(
    output_dir: str, src_lang: str, tgt_lang: str, decoded_preds: List[str]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"test-{src_lang}-{tgt_lang}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(f"{pred}\n" for pred in decoded_preds)


def process_translations(
    model: str, client: OpenAI, input_file: str, src_lang: str, tgt_lang: str
) -> List[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        sources = [line.strip() for line in f]

    decoded_preds = []
    for source in tqdm(sources, desc=f"Translating {src_lang} to {tgt_lang}"):
        output = translate_text(
            model, client, source.strip(), LANG_DICT[src_lang], LANG_DICT[tgt_lang]
        )
        decoded_preds.append(output)

    return decoded_preds


def main(args: argparse.Namespace) -> None:
    client = OpenAI()

    if args.direction == "en2x":
        src_lang = "en"
        for tgt_lang in ["fr", "de", "es", "it"]:
            input_file = os.path.join(args.input_dir, "en.devtest")
            decoded_preds = process_translations(
                args.model, client, input_file, src_lang, tgt_lang
            )
            save_results(args.output_dir, src_lang, tgt_lang, decoded_preds)
    elif args.direction == "x2en":
        tgt_lang = "en"
        for src_lang in ["fr", "de", "es", "it"]:
            input_file = os.path.join(args.input_dir, f"{src_lang}.devtest")
            decoded_preds = process_translations(
                args.model, client, input_file, src_lang, tgt_lang
            )
            save_results(args.output_dir, src_lang, tgt_lang, decoded_preds)
    else:
        raise ValueError(f"Invalid direction: {args.direction}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
