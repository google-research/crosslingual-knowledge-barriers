# Harry Potter Quiz evaluation

This folder contains scripts for reproducing our results on in-domain crosslingual knowledge barriers, specifically the performance on Harry Potter-related QAs.

## Dataset

The corresponding authors have manually curated a multiple-choice question-answering dataset called the Harry Potter Quiz (HP-Quiz, [HuggingFace link](https://huggingface.co/datasets/cross-ling-know/HarryPotter-Quiz)). The dataset is sourced from [Harry Potter Wiki](https://harrypotter.fandom.com/wiki/Main_Page) pages and consists of 300 questions: 157 about characters and 143 about magic spells. Each question is formatted as multiple-choice and presented in five languages: English, French, German, Italian, and Spanish.

We use this dataset in our evaluation.

## Models

The corresponding authors release the Llama2-7B and Llama3-8B series models used in our evaluation, fine-tuned on English-only corpora or on mixed-translated corpora, including:

- The models trained on the English version of a general corpus (i.e., WikiText-2): [Llama2-7B](https://huggingface.co/cross-ling-know/llama2-7b-wiki2-en), [Llama3-8B](https://huggingface.co/cross-ling-know/llama3-8b-wiki2-en)
- The models trained on a mixed translated version of a general corpus (i.e., WikiText-2), with the translation unit being per document: [Llama2-7B](https://huggingface.co/cross-ling-know/llama2-7b-wiki2-mixed-lang-document), [Llama3-8B](https://huggingface.co/cross-ling-know/llama3-8b-wiki2-mixed-lang-document)
- The models trained on a mixed translated version of a general corpus (i.e., WikiText-2), with the translation unit being per sentence: [Llama2-7B](https://huggingface.co/cross-ling-know/llama2-7b-wiki2-mixed-lang-sentence), [Llama3-8B](https://huggingface.co/cross-ling-know/llama3-8b-wiki2-mixed-lang-sentence)
- The models trained on a mixed translated version of a general corpus (i.e., WikiText-2), with the translation unit being 8 words: [Llama2-7B](https://huggingface.co/cross-ling-know/llama2-7b-wiki2-mixed-lang-sentence8words), [Llama3-8B](https://huggingface.co/cross-ling-know/llama3-8b-wiki2-mixed-lang-sentence8words)

## Evaluation

Please follow the instruction provided in the [main README](../README.md) to set up the environemt.

The evaluation can be performed by running the `evaluate.py` script:

```bash
python evaluate.py --model_name MODEL_NAME --lang LANGUAGE_CODE --save_dir OUTPUT_DIR
```

where LANGUAGE_CODE is a value from {en, fr, de, it, es}.

After the evaluation, the per-question results are saved as a CSV file in your designated `OUTPUT_DIR`.

## Wiki103-HP

This folder `Wiki103-HP` provides resources for processing Harry Potter (HP)-related documents within the Wikitext-103 dataset.
Specifically,

- `Wiki103-HP/HP_related_index_for_wikitext-103.json`: A JSON file containing the indices of HP-related documents in Wikitext-103.
- `Wiki103-HP/read_index_for_wikitext-103.ipynb`: A Jupyter Notebook for processing the Wikitext-103 dataset to extract HP-related and HP-unrelated subsets.
  We use the HP-related subset as domain-specifc corpus to finetune LLMs.
