import os
from functools import partial
import random
import string
import numpy as np
from loguru import logger

from clearml import Logger, Task
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    TrainingArguments,
    Trainer,
)
from transformers.data.data_collator import default_data_collator
from datasets import (
    Features,
    Sequence,
    Value,
    Array2D,
    Array3D,
    load_dataset,
    load_from_disk,
)
import evaluate

from utils import setup_dataset, get_random_str


def get_label2id_map(dataset):
    label2id, id2label = {}, {}
    label_feature = dataset["test"].features["ner_tags"].feature
    for name in label_feature.names:
        label2id[name] = label_feature.str2int(name)
        id2label[label_feature.str2int(name)] = name

    return label2id, id2label


def compute_metrics_callback(p, id2label):
    evaluator = evaluate.load("seqeval")

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Remove predicitons for tokens of class other
    fields_predictions = [
        [p for (p, l) in zip(prediction, label) if l != "O"]
        for prediction, label in zip(true_predictions, true_labels)
    ]
    fields_labels = [
        [l for (p, l) in zip(prediction, label) if l != "O"]
        for prediction, label in zip(true_predictions, true_labels)
    ]

    results = evaluator.compute(predictions=true_predictions, references=true_labels)
    results_fields = evaluator.compute(
        predictions=fields_predictions, references=fields_labels
    )

    metrics = {}
    for tag, res in zip(["all", "fields"], [results, results_fields]):
        for m in ["precision", "recall", "f1", "accuracy"]:
            metrics[m + "_" + tag] = res[f"overall_{m}"]

    return metrics


def train(config):
    exp_name = "train-" + get_random_str(n=6)
    exp_folder = os.path.join(config["output_folder"], exp_name)
    os.makedirs(exp_folder)

    os.environ.setdefault("CLEARML_LOG_MODEL", "False")
    task = Task.init(
        project_name=config["project"],
        task_name=exp_name,
        tags=["train"],
    )
    task.connect(config)

    dataset = setup_dataset(config["dataset"])
    label2id, id2label = get_label2id_map(dataset)

    processor = LayoutLMv3Processor.from_pretrained(
        "microsoft/layoutlmv3-base", apply_ocr=False
    )
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base", label2id=label2id, id2label=id2label
    )
    model.compile()

    features = Features(
        {
            "pixel_values": Array3D(dtype="float32", shape=(3, 224, 224)),
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),  # why 512????
            "labels": Sequence(feature=Value(dtype="int64")),
        }
    )

    dataset = dataset.map(
        lambda samples: processor(
            samples["image"],
            samples["words"],
            boxes=samples["bboxes"],
            word_labels=samples["ner_tags"],
            truncation=True,
            padding="max_length",
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        features=features,
    )

    train_dataset, test_dataset = dataset["train"], dataset["test"]
    val_dataset = None if config["val_split"] is None else dataset[config["val_split"]]

    train_dataset.set_format("pytorch")

    training_args = TrainingArguments(
        output_dir=exp_folder,
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch"],
        per_device_eval_batch_size=config["batch"],
        learning_rate=config["lr"],
        eval_strategy=None if val_dataset is None else "steps",
        eval_steps=config["val_steps"],
        logging_steps=config["log_steps"],
        load_best_model_at_end=False,
        save_steps=config["checkpoint_steps"],
        fp16=config["mixed_precision"] == "fp16",
        tf32=config["mixed_precision"] == "tf32",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor,
        data_collator=default_data_collator,
        compute_metrics=partial(compute_metrics_callback, id2label=id2label),
    )

    trainer.train()

    test_metrics = trainer.predict(test_dataset).metrics
    print(f"Test metrics:")
    for n, v in test_metrics.items():
        for m in ["f1", "recall", "precision", "accuracy"]:
            if m in n:
                print(f"\t{n}: {v}")
                Logger.current_logger().report_scalar(
                    "test_all" if "all" in n else "test_fields",
                    n,
                    iteration=0,
                    value=v,
                )

    last_ckpt = os.path.join(exp_folder, "last")
    trainer.save_model(last_ckpt)
    if config["upload_model"]:
        logger.info("Saving last model checkpoint to ClearML, this may take a while...")
        task.upload_artifact("model", last_ckpt)
        logger.info("Upload done!")


if __name__ == "__main__":
    from argparse import ArgumentParser
    import yaml

    parser = ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the config file for training", required=True
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_args = yaml.load(f, Loader=yaml.FullLoader)

    train(config=config_args)
