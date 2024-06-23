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

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm

LANG_DICT: Dict[str, str] = {
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "en": "eng_Latn",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translation script using NLLB model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="facebook/nllb-200-3.3B",
        help="Model name on HuggingFace or the path to local model",
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
        default="outputs/nllb-3.3b",
        help="Directory to save translation results",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="en2x",
        choices=["x2en", "en2x"],
        help="Translation direction: 'x2en' for X to English, 'en2x' for English to X",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for translation",
    )
    return parser.parse_args()


def save_results(
    output_dir: str, src_lang: str, tgt_lang: str, decoded_preds: List[str]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, f"test-{src_lang}-{tgt_lang}"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("\n".join(decoded_preds))


def translate_batch(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    sources: List[str],
    tgt_lang: str,
    device: torch.device,
) -> List[str]:
    inputs = tokenizer(
        sources, return_tensors="pt", padding=True, truncation=True, max_length=256
    ).to(device)
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[LANG_DICT[tgt_lang]],
        max_length=256,
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    if args.direction == "en2x":
        src_lang, tgt_langs = "en", ["fr", "de", "es", "it"]
    else:
        src_langs, tgt_lang = ["fr", "de", "es", "it"], "en"

    for src_lang in [src_lang] if args.direction == "en2x" else src_langs:
        for tgt_lang in tgt_langs if args.direction == "en2x" else [tgt_lang]:
            input_file = os.path.join(args.input_dir, f"{src_lang}.devtest")

            with open(input_file, "r", encoding="utf-8") as f:
                sources = [line.strip() for line in f]

            decoded_preds = []
            for i in tqdm(
                range(0, len(sources), args.batch_size),
                desc=f"{src_lang} to {tgt_lang}",
            ):
                batch = sources[i : i + args.batch_size]
                decoded_preds.extend(
                    translate_batch(model, tokenizer, batch, tgt_lang, device)
                )

            save_results(args.output_dir, src_lang, tgt_lang, decoded_preds)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
