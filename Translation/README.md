# Translation Evaluation

This folder contains scripts for reproducing our translation evaluation results. We focus on five widely spoken languages: English (en), French (fr), German (de), Spanish (es), and Italian (it).

## Dataset

We evaluate translation performance on a subset of 100 examples from the [FLoRes-101 benchmark](https://arxiv.org/abs/2106.03193).
This subset is provided in the `data` folder for convenience:

```
data/
└── flores
    ├── de.devtest
    ├── en.devtest
    ├── es.devtest
    ├── fr.devtest
    └── it.devtest
```

Each row across different files represents the same sentence in different languages.

## Models

We evaluate three categories of models:

- Open-source LLMs with strong multilingual performance:
  - [Llama2-7B](https://huggingface.co/meta-llama/Llama-2-7b)
  - [Llama2-13B](https://huggingface.co/meta-llama/Llama-2-13b)
  - [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
  - [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
- Proprietary models (via APIs): [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5-turbo) and [GPT-4](https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4)
- Baselines:
  - [NLLB-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) model: The largest supervised encoder-decoder translation model from the NLLB family, trained on a parallel corpus from various sources for 204 languages.
  - Internal Google Translate API

## Translation

### 1. Open-source LLMs

For open-source LLMs, we use the `run_llmmt.py` script from [the ALMA repository](https://github.com/fe1ixxu/ALMA) to run translations. We update the `--model_name_or_path` parameter to the corresponding evaluated models and set the `--text_test_file` parameter to the FLoRes data folder.

### 2. Proprietary models

For proprietary models, you need to set the `OPENAI_API_KEY` environment variable.

```bash
pip install openai
export OPENAI_API_KEY=YOUR_OPENAI_KEY
```

Use the following scripts to run translations with these models. Here is an example for GPT-4:

```bash
python translate_gpt.py --model gpt-4 --direction en2x --output_dir OUTPUT_DIR # Translate from English to another language
python translate_gpt.py --model gpt-4 --direction x2en --output_dir OUTPUT_DIR # Translate from another language to English
```

The outputs will be saved under `OUTPUT_DIR`.

### 3. The NLLB model

Run the following script to use the NLLB model for translation:

```bash
python translate_nllb.py --direction en2x  --output_dir OUTPUT_DIR # Translate from English to another language
python translate_nllb.py --direction x2en  --output_dir OUTPUT_DIR # Translate from another language to English
```

The outputs will be saved under `OUTPUT_DIR`.

## Evaluation

We use [the COMET score](https://github.com/Unbabel/COMET) to evaluate the translation performance. Install COMET by running:

```bash
pip install unbabel-comet
```

Please refer to [this example script](https://github.com/fe1ixxu/ALMA/blob/master/evals/eval_generation.sh) from the ALMA repository to calculate COMET socres based on translation results.

## Results

For reference, we provide sample translation results and COMET scores (for 50 examples) for each evaluated model in the `outputs` folder:

```
outputs
├── google-translate
├── gpt3.5
├── gpt4
├── llama2-13b
├── llama2-7b
├── llama3-8b
├── mistral-7b
└── nllb-3.3b
```
