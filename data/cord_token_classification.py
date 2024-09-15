import json
import datasets

from data.utils import normalize_bbox

_CITATION = """\
@article{park2019cord,
  title={CORD: A Consolidated Receipt Dataset for Post-OCR Parsing},
  author={Park, Seunghyun and Shin, Seung and Lee, Bado and Lee, Junyeop and Surh, Jaeheung and Seo, Minjoon and Lee, Hwalsuk}
  booktitle={Document Intelligence Workshop at Neural Information Processing Systems}
  year={2019}
}
"""

_DESCRIPTION = """
A modified version of the original Cord-v2 dataset that supports both structured output generation and token classification as learning modalities.
"""


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


class CordTokenClassification(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="cord-v2-token-classification",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            dataset_name="cord-v2-token-classification",
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "is_key": datasets.Sequence(datasets.Value("bool")),
                    "bboxes": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int64"))
                    ),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=[
                                "O",
                                "MENU.NM",
                                "MENU.NUM",
                                "MENU.UNITPRICE",
                                "MENU.CNT",
                                "MENU.DISCOUNTPRICE",
                                "MENU.PRICE",
                                "MENU.ITEMSUBTOTAL",
                                "MENU.VATYN",
                                "MENU.ETC",
                                "MENU.SUB.NM",
                                "MENU.SUB.UNITPRICE",
                                "MENU.SUB.CNT",
                                "MENU.SUB.PRICE",
                                "MENU.SUB.ETC",
                                "VOID_MENU.NM",
                                "VOID_MENU.PRICE",
                                "SUB_TOTAL.SUBTOTAL_PRICE",
                                "SUB_TOTAL.DISCOUNT_PRICE",
                                "SUB_TOTAL.SERVICE_PRICE",
                                "SUB_TOTAL.OTHERSVC_PRICE",
                                "SUB_TOTAL.TAX_PRICE",
                                "SUB_TOTAL.ETC",
                                "TOTAL.TOTAL_PRICE",
                                "TOTAL.TOTAL_ETC",
                                "TOTAL.CASHPRICE",
                                "TOTAL.CHANGEPRICE",
                                "TOTAL.CREDITCARDPRICE",
                                "TOTAL.EMONEYPRICE",
                                "TOTAL.MENUTYPE_CNT",
                                "TOTAL.MENUQTY_CNT",
                            ]
                        )
                    ),
                    "image": datasets.features.Image(),
                    "gt_parse": datasets.Value("string"),
                }
            ),
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        cordv2 = datasets.load_dataset("naver-clova-ix/cord-v2")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"original_dataset": cordv2["train"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"original_dataset": cordv2["validation"]},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"original_dataset": cordv2["test"]},
            ),
        ]

    # Disable segment level layout for fairer comparison
    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, original_dataset):
        """
        Takes the raw CORD dataset and builds samples for training and evaluation in the format expected by Donut and LayoutLMv3.
        For each document sample the following are provided:
        - list of words
        - list of bbox of each word
        - list of token classification labels, i.e. a label for each word
        - document image
        - list of booleans defining whether each word corresponds to a key in a form; Donut is only trained to extract non-key information, so this data is useful for filtering the predictions made by LayoutLMv3 to ignore key-text and compare LayoutLMv3 and Donut on the same data
        - dict containing the relevant information to extract from the document in a structured format;

        LayoutLMv3 and Donut are evaluated on two metrics:
        - F1 accuracy between the words extracted by the model with their predicted class and ground truth; measures the ability to recover and recognize information in the document
        - TED between the extracted information and the expected structured ground truth; measures the ability to recover information in the document and its structure.

        The original CORD dataset defined a specific structure for the information contained in any given receipt. Each word in the document has a super and sub category. Each super-category can have multiple instance of the same sub-category, but they are divided within groups such that each group contains at most one instance of each sub-category. This is clear with an example: the 'menu' super-category can include multiple items, each item is assigned to a group with its own sub-labels such as 'cnt', 'price', 'nm', etc.

        Example:

        {'menu': [{'cnt': '1x', 'unitprice': '36.000', 'price': '36.000', 'nm': 'Kimchi P'}, {'cnt': '1x', 'unitprice': '0.000', 'price': 'Rp 0.000', 'nm': 'Fre ice grentea'}], 'total': {'total_price': '36.000', 'cashprice': 'Rp 51.000', 'changeprice': '15.000'}}



        The way in which this structure was created for the raw data was not documented in the Donut paper code, but I managed to reverse engineer it to get the same exact result. The main reason for why I did this was to make LayoutLMv3 extract that with this precise in format in order to be able to compare it with Donut.
        Nativelky LayoutLMv3 only predicts the label, what it does not predict is the group to which a word belongs to, i.e. if we have three words classified as menu.price we cannot know if they blong to the same of different groups. It turns out however that data groups are contiguous if scanning the document in reading order, so it was enough to slightly modify LayoutLMv3 target by appending "B-" to the label for tokens at the beggining of a group and "I-" for those inside it. Words to corresponding to the key of a field are assigned a target label of 'other', following the convention of Donut, in which these data is not extracted. Natively, LayoutLMv3 used this to distinguish the beginning of a new line. With these changes, LayoutLMv3 predictions contains all information needed to format the extracted data with the same structure predicted by Donut.



        Here the menu has two items, so `menu` has two data-groups, one for each item.

        """
        for guid, sample in enumerate(original_dataset):
            image = sample["image"]
            d = json.loads(sample["ground_truth"])
            gt_parse_original = d["gt_parse"]

            words = []
            is_key = []
            bboxes = []
            ner_tags = []
            words_data = []
            for item in d["valid_line"]:
                for word in item["words"]:
                    words.append(word["text"])
                    is_key.append(word["is_key"])
                    bboxes.append(normalize_bbox(quad_to_box(word["quad"]), image.size))

                    if word["is_key"]:
                        ner_tags.append("O")
                        continue
                    ner_tags.append("O" if word["is_key"] else item["category"].upper())

            yield guid, {
                "id": str(guid),
                "words": words,
                "is_key": is_key,
                "bboxes": bboxes,
                "ner_tags": ner_tags,
                "image": image,
                "gt_parse": json.dumps(gt_parse_original),
            }
