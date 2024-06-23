# Crosslingual MMLU Dataset

This dataset is an extension of the original MMLU dataset with translations into multiple languages. The folder structure and contents are as follows:
```
data/
├── test/
│ ├── abstract_algebra_test.csv
│ ├── anatomy_test.csv
│ └── ...
├── dev/
│ ├── abstract_algebra_dev.csv
│ ├── anatomy_dev.csv
│ └── ...
├── test_full_de/
│ ├── abstract_algebra_test.csv
│ ├── anatomy_test.csv
│ └── ...
├── dev_full_de/
│ ├── abstract_algebra_dev.csv
│ ├── anatomy_dev.csv
│ └── ...
├── test_mixup/
│ ├── abstract_algebra_test.csv
│ ├── anatomy_test.csv
│ └── ...
├── dev_mixup/
│ ├── abstract_algebra_dev.csv
│ ├── anatomy_dev.csv
│ └── ...
└── ...
````
Specifically, 
- `data/`
  - `test/`: Original MMLU test data
  - `dev/`: Original MMLU dev data
  - `test_{mode}_{language}/`: Translated test data for various modes and languages
  - `dev_{mode}_{language}/`: Translated dev data for various modes and languages
  - `test_mixup/`: Mixed-language test data
  - `dev_mixup/`: Mixed-language dev data


### Modes

- `full`: Complete translation of both questions and options.
- `options`: Only options are translated.
- `question`: Only question is translated.
- `gt`: Ground truth option is translated.
- `gt_question`: Question and ground truth option are translated.
- `onewrong`: One wrong option translated.
- `threewrong`: Three wrong options translated.

### Languages

- `de`: German
- `es`: Spanish
- `fr`: French
- `it`: Italian


## Usage

To use the dataset, navigate to the appropriate folder for your desired mode and language. The files are named according to the original MMLU dataset with the same naming convention but within the specified mode and language folders.
