from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
from shapely.geometry import Polygon
import numpy as np


def calculate_iou(box_1: List[float], box_2: List[float]) -> float:
    def map_bbox(box):
        (x1, y1, x2, y2) = box
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    _box_1 = map_bbox(box_1)
    _box_2 = map_bbox(box_2)

    poly_1 = Polygon(_box_1)
    poly_2 = Polygon(_box_2)

    iou = poly_1.intersection(poly_2).area
    min_area = min(poly_1.area, poly_2.area)
    return iou / min_area


def norm_box(box, size):
    w, h = size
    return (
        np.array([box[0][0] / w, box[0][1] / h, box[2][0] / w, box[2][1] / h]) * 1000
    ).astype(np.int32)


@dataclass
class InferenceSample:
    image: str
    words: str
    boxes: str


class InferenceEngine:
    def __init__(self, checkpoint, device):
        self.device = device
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(checkpoint).to(
            device
        )
        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base", apply_ocr=False
        )

    def run(self, sample: Dict, ocr: bool = False):
        encoding = self.processor(
            sample["image"],
            sample["words"],
            boxes=sample["bboxes"],
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
        )
        offset_mapping = encoding.pop("offset_mapping").squeeze()

        for k, v in encoding.items():
            encoding[k] = v.to(self.device)

        with torch.no_grad():
            outputs = self.model(**encoding)

        preds_i_raw = outputs.logits.argmax(-1).squeeze().tolist()

        words, label_preds, field_pred = [], [], {}
        for i in range(len(offset_mapping)):
            # Skip tokens inside a word
            if offset_mapping[i][0] != 0:
                continue

            label_pred = self.model.config.id2label[preds_i_raw[i]]

            # Skip 'other' words and special tokens
            if encoding.word_ids()[i] is None:
                continue

            word_idx = encoding.word_ids()[i]

            words.append(sample["words"][word_idx])
            label_preds.append(label_pred)

            if label_pred == "O":
                continue

        return words, label_preds
