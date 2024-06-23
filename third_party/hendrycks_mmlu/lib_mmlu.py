# This file is modified from an open source repo: https://github.com/hendrycks/test/blob/master/evaluate.py

import json
import os
import numpy as np
import pandas as pd


def eval_subject(subject, llm, dev_df, test_df, mmlu_proc, ntrain=0, choices=["A", "B", "C", "D"]):

    cors = []
    all_probs = []
    all_prompts = []
    all_predicts = []
    all_labels = []

    for i in range(test_df.shape[0]):
        # get prompt and make sure it fits
        prompt, label = mmlu_proc.gen_test_prompt(
            ntrain, test_df, dev_df, i, subject)
        pred, probs = llm.query(prompt, choices)

        cor = pred == label

        if i % 20 == 0:
            print(prompt)
            print("\npredict\n", pred)
            print("\nlabel\n", label)

        cors.append(cor)
        all_probs.append(probs)
        all_predicts.append(pred)
        all_prompts.append(prompt)
        all_labels.append(label)

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs, all_predicts, all_prompts, all_labels


def read_corrs(output_filename):
    with open(output_filename, 'r') as file:
        # Read each line of the file
        cors = []
        for line in file:
            # Parse the JSON data from each line
            data = json.loads(line)
            cors.append(data["correct"])

    cors = np.array(cors)
    return cors


def eval_subjects(subjects, dev_folder, test_folder, llm, mmlu_proc, output_folder, ntrain=0, choices=["A", "B", "C", "D"], infer_mode="logits"):

    all_cors = []
    results = {"subject": {}}

    for subject in subjects:
        test_output_file = os.path.join(
            output_folder, "{}.jsonl".format(subject))
        if os.path.exists(test_output_file):
            try:
                cors = read_corrs(test_output_file)
                all_cors.append(cors)
                continue
            except Exception as e:
                print(subject, e)

        if ntrain > 0:
            dev_df = pd.read_csv(os.path.join(
                dev_folder, subject + "_dev.csv"), header=None)[: ntrain]
        else:
            dev_df = None

        test_df = pd.read_csv(os.path.join(
            test_folder, subject + "_test.csv"), header=None)

        cors, acc, probs, predicts, prompts, labels = eval_subject(
            subject, llm, dev_df, test_df, mmlu_proc, ntrain, choices)

        all_cors.append(cors)

        test_df["predicts"] = predicts
        test_df["labels"] = labels
        test_df["correct"] = cors
        test_df["prompts"] = prompts

        if infer_mode == "logits":
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["choice{}_probs".format(choice)] = probs[:, j]

        test_df.to_json(test_output_file, orient='records', lines=True)
        results["subject"][subject] = acc

    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    return results


def eval_subjects_hf_dataset(subjects, hf_dataset_name, llm, mmlu_proc, output_folder, ntrain=0, choices=["A", "B", "C", "D"], infer_mode="logits"):

    from datasets import load_dataset

    all_cors = []
    results = {"subject": {}}

    for subject in subjects:
        test_output_file = os.path.join(
            output_folder, "{}.jsonl".format(subject))
        if os.path.exists(test_output_file):
            try:
                cors = read_corrs(test_output_file)
                all_cors.append(cors)
                continue
            except Exception as e:
                print(subject, e)

        if ntrain > 0:
            dev_df = load_dataset(hf_dataset_name, data_files=[
                                  f"data/dev/{subject}_dev.csv"])['train'].to_pandas()[: ntrain]
        else:
            dev_df = None

        test_df = load_dataset(hf_dataset_name, data_files=[
                               f"data/test/{subject}_test.csv"])['train'].to_pandas()

        cors, acc, probs, predicts, prompts, labels = eval_subject(
            subject, llm, dev_df, test_df, mmlu_proc, ntrain, choices)

        all_cors.append(cors)

        test_df["predicts"] = predicts
        test_df["labels"] = labels
        test_df["correct"] = cors
        test_df["prompts"] = prompts

        if infer_mode == "logits":
            for j in range(probs.shape[1]):
                choice = choices[j]
                test_df["choice{}_probs".format(choice)] = probs[:, j]

        test_df.to_json(test_output_file, orient='records', lines=True)
        results["subject"][subject] = acc

    weighted_acc = np.mean(np.concatenate(all_cors))
    results["weighted_accuracy"] = weighted_acc
    print("Average accuracy: {:.3f}".format(weighted_acc))

    return results
