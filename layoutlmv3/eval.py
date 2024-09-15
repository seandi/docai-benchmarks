from argparse import ArgumentParser
from layoutlmv3.train import setup_dataset
import yaml
import json
from clearml import Task
from tqdm import tqdm
import numpy as np
from layoutlmv3.inference import InferenceEngine
from utils import JSONParseEvaluator


def eval(configs):
    dataset = setup_dataset(configs["dataset"])
    inference = InferenceEngine(checkpoint=configs["checkpoint"], device="cuda")
    total_tp, total_fn_or_fp = 0, 0

    for sample in tqdm(dataset[configs['split']]):

        words, preds = inference.run(sample, ocr=False)

        token_gt = [
            (
                sample["words"][i],
                inference.model.config.id2label[sample["ner_tags"][i]].lower(),
            )
            for i in range(len(sample["words"]))
            if not sample["is_key"][i]
        ]

        token_preds = []
        for i, word in enumerate(words):
            if preds[i] == "O":
                continue
            token_preds.append((word, preds[i].lower()))

        # Evaluate word recovery and classification with f1 acc
        for p in token_preds:
            if p in token_gt:
                total_tp += 1
                token_gt.remove(p)
            else:
                total_fn_or_fp += 1
        total_fn_or_fp += len(token_gt)

    f1_acc = total_tp / (total_tp + total_fn_or_fp / 2) * 100
    print(f"F1 acc: {f1_acc}% TED acc: n.a.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the config file for training", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    eval(configs)
