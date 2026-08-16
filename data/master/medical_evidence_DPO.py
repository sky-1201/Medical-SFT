# coding=utf-8

# Lint as: python3
"""Medical Evidence DPO Dataset for Direct Preference Optimization"""


import json

import datasets


_CITATION = """\
@misc{medical_evidence_dpo,
  title        = {Medical Evidence DPO Dataset},
  author       = {Medical Evidence Team},
  year         = {2024},
  description  = {A Chinese medical Q&A preference dataset for training language models with Direct Preference Optimization (DPO).}
}
"""

_DESCRIPTION = """\
Medical Evidence DPO Dataset is a Chinese medical question-answer preference dataset designed for
Direct Preference Optimization (DPO) training. Each sample contains a medical question along with
a chosen (high-quality) answer and a rejected (lower-quality) answer, enabling language models
to learn preference alignment in the medical domain.

The dataset covers various medical topics including:
- Clinical guidelines and treatment management
- Disease mechanisms and pathophysiology
- Drug therapy and pharmacology
- Medical diagnostics and laboratory interpretation
- Medical research methodology and evidence evaluation
"""

_HOMEPAGE = "https://modelscope.cn/datasets/your-org/medical-evidence-dpo"
_LICENSE = "cc-by-nc-sa-4.0"


class MedicalEvidenceDPO(datasets.GeneratorBasedBuilder):
    """Medical Evidence DPO Dataset"""

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="medical_evidence_dpo",
            version=datasets.Version("1.0.0"),
            description=_DESCRIPTION,
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "prompt": datasets.Value("string"),
                    "chosen": datasets.Value("string"),
                    "rejected": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[
                datasets.TaskTemplate(
                    task="preference-alignment",
                    prompt_column="prompt",
                    chosen_column="chosen",
                    rejected_column="rejected",
                )
            ],
        )

    def _split_generators(self, dl_manager):
        data_file = dl_manager.download_and_extract("dpo_answer.jsonl")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_file, "split": "train"},
            ),
        ]

    def _generate_examples(self, filepath, split):
        """Generate examples from the DPO dataset."""
        with open(filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line.strip())
                yield idx, {
                    "prompt": data.get("prompt", ""),
                    "chosen": data.get("chosen", ""),
                    "rejected": data.get("rejected", ""),
                }