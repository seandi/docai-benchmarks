import argparse
from tqdm import tqdm
import json
import numpy as np
import torch
from transformers import AutoModel

from utils import JSONParseEvaluator, setup_dataset

from transformers import DonutProcessor, VisionEncoderDecoderModel
import re


def load_pretrained_model(checkpoint):
    processor = DonutProcessor.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2"
    )
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint)

    if torch.cuda.is_available():
        model.half()
        model.to("cuda")

    model.eval()

    return model, processor


class DonutInference:
    def __init__(self, config) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.processor = load_pretrained_model(config["checkpoint"])
    
    def run(self, sample):
        pixel_values = self.processor(
            sample["image"], return_tensors="pt"
        ).pixel_values
        
        if self.device == "cuda":
            pixel_values = pixel_values.half()
            

        # prepare decoder inputs
        task_prompt = f"<s_cord-v2>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.decoder.config.max_position_embeddings,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = self.processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
            self.processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(
            r"<.*?>", "", sequence, count=1
        ).strip()  # remove first task start token

        return self.processor.token2json(sequence)


def eval(config):
    testset = setup_dataset(config["dataset"])["test"]
    id2label = testset.features["ner_tags"].feature.int2str

    inference = DonutInference(config)
    evaluator = JSONParseEvaluator()

    total_tp, total_fn_or_fp, ted_acc = 0, 0, []
    for _, sample in tqdm(enumerate(testset), total=len(testset)):
        fields_pred = inference.run(sample)

        ted_acc.append(
            evaluator.cal_acc(
                fields_pred,
                json.loads(sample["gt_parse"]),
            )
        )

        token_preds = []
        for pred, value in evaluator.flatten(evaluator.normalize_dict(fields_pred)):
            token_preds.extend([(word, pred) for word in value.split(" ")])

        token_gt = []
        for i, word in enumerate(sample["words"]):
            if not sample["is_key"][i]:
                token_gt.append((word, id2label(sample["ner_tags"][i]).lower()))

        for p in token_preds:
            if p in token_gt:
                total_tp += 1
                token_gt.remove(p)
            else:
                total_fn_or_fp += 1
        total_fn_or_fp += len(token_gt)

    f1_acc = total_tp / (total_tp + total_fn_or_fp / 2) * 100.0
    ted_acc = np.mean(ted_acc) * 100.0
    print(f"F1 acc: {f1_acc}% TED acc: {ted_acc}%")


if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, left_argv = parser.parse_known_args()

    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    eval(config=configs)
