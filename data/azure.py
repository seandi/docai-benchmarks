import base64
import os
from dotenv import load_dotenv
from io import BytesIO
from typing import List
from shapely.geometry import Polygon
from functools import partial

import datasets
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import (
    AnalyzeDocumentRequest,
    AnalyzeResult,
)

from utils import setup_dataset
from data.utils import normalize_bbox


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

    if min_area == 0:
        return 0
    return iou / min_area


def apply_azure_ocr(sample, client):
    buffer = BytesIO()
    sample["image"].save(buffer, format="png")
    doc_b64 = base64.b64encode(buffer.getvalue()).decode()

    poller = client.begin_analyze_document(
        "prebuilt-read",
        analyze_request=AnalyzeDocumentRequest(bytes_source=doc_b64),
    )
    result: AnalyzeResult = poller.result()

    words = []
    boxes = []
    is_key = []
    labels = []
    for page in result.get("pages", []):
        w, h = page["width"], page["height"]
        for word in page.get("words", []):

            box = word["polygon"]
            box = [box[0], box[1], box[4], box[5]]
            box = normalize_bbox(box, (w, h))

            match = -1
            for i, gt_box in enumerate(sample["bboxes"]):
                if calculate_iou(box, gt_box) > 0.8:
                    match = i
                    break

            words.append(word["content"])
            boxes.append(box)
            if match == -1:
                is_key.append(False)
                labels.append(0)

            else:
                is_key.append(sample["is_key"][i])
                labels.append(sample["ner_tags"][i])

    sample["words"] = words
    sample["bboxes"] = boxes
    sample["is_key"] = is_key
    sample["ner_tags"] = labels

    return sample


def build_azure_testset(dataset):
    load_dotenv()

    client = DocumentIntelligenceClient(
        endpoint=os.environ["ENDPOINT"],
        credential=AzureKeyCredential(os.environ["AZURE_DOCAI_KEY"]),
    )

    testset_azure = dataset["test"].map(
        partial(apply_azure_ocr, client=client)
    )
    dataset["testazure"] = testset_azure

    return dataset
