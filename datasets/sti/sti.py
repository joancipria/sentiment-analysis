# coding=utf-8
# Copyright 2020 HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""STI dataset"""


import csv
import os

from datasets.tasks import TextClassification

import datasets

_DESCRIPTION = """\
STI emotion dataset
"""

_CITATION = """\
@inproceedings{moreno2022conversational,
  title={A Conversational Agent for Medical Disclosure of Sexually Transmitted Infections},
  author={Moreno, Joan C and S{\'a}nchez-Anguix, Victor and Alberola, Juan M and Juli{\'a}n, Vicente and Botti, Vicent},
  booktitle={International Conference on Hybrid Artificial Intelligence Systems},
  pages={431--442},
  year={2022},
  organization={Springer}
}
"""

_CLASS_NAMES = ["anger", "fear", "sadness", "joy", "disgust", "surprise", "others"]

_HOMEPAGE = "https://github.com/joancipria/sentiment-analysis/blob/master/"

_URLS = {
    "es": "https://raw.githubusercontent.com/joancipria/sentiment-analysis/sti-dataset/datasets/sti/splits/es/",
    "en": "https://raw.githubusercontent.com/joancipria/sentiment-analysis/sti-dataset/datasets/sti/splits/en/",
}


class EmoEvent(datasets.GeneratorBasedBuilder):
    """EmoEvent classification dataset."""

    VERSION = datasets.Version("1.1.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="es", version=VERSION, description="This part of my dataset covers the english version"
        ),
        datasets.BuilderConfig(
            name="en", version=VERSION, description="This part of my dataset covers the spanish version"
        ),
    ]

    DEFAULT_CONFIG_NAME = "es"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "event": datasets.Value("string"),
                    "tweet": datasets.Value("string"),
                    "offensive": datasets.Value("string"),
                    "label": datasets.features.ClassLabel(
                        names=["anger", "fear", "sadness", "joy", "disgust", "surprise", "others"]
                    ),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[TextClassification(text_column="tweet", label_column="label")],
        )

    def _split_generators(self, dl_manager):

        urls = _URLS[self.config.name]
        _TRAIN_DOWNLOAD_URL = os.path.join(urls, "train.tsv")
        _DEV_DOWNLOAD_URL = os.path.join(urls, "dev.tsv")
        _TEST_DOWNLOAD_URL = os.path.join(urls, "test.tsv")

        print(_TRAIN_DOWNLOAD_URL)

        train_path = dl_manager.download_and_extract(_TRAIN_DOWNLOAD_URL)
        dev_path = dl_manager.download_and_extract(_DEV_DOWNLOAD_URL)
        test_path = dl_manager.download_and_extract(_TEST_DOWNLOAD_URL)
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": dev_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """Generate AG News examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter="\t", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            next(csv_reader)
            for id_, row in enumerate(csv_reader):
                id_tweet, event, tweet, offensive, label = row
                yield id_, {"id": id_tweet, "event": event, "tweet": tweet, "offensive": offensive, "label": label}
