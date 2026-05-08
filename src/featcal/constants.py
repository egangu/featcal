from __future__ import annotations

from dataclasses import dataclass


TASKS = (
    "sun397",
    "stanford-cars",
    "resisc45",
    "eurosat",
    "svhn",
    "gtsrb",
    "mnist",
    "dtd",
)

BASE_MODEL_ID = "openai/clip-vit-base-patch32"
EXPERT_MODEL_IDS = {
    task: f"tanganke/clip-vit-base-patch32_{task}" for task in TASKS
}


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    name: str | None = None
    train_split: str = "train"
    test_split: str = "test"


DATASET_SPECS = {
    "sun397": DatasetSpec("tanganke/sun397"),
    "stanford-cars": DatasetSpec("tanganke/stanford_cars"),
    "resisc45": DatasetSpec("tanganke/resisc45"),
    "eurosat": DatasetSpec("tanganke/eurosat"),
    "svhn": DatasetSpec("svhn", "cropped_digits"),
    "gtsrb": DatasetSpec("tanganke/gtsrb"),
    "mnist": DatasetSpec("ylecun/mnist"),
    "dtd": DatasetSpec("tanganke/dtd"),
}

REFERENCE_TA_ACCURACY = {
    "sun397": 0.570125937461853,
    "stanford-cars": 0.5570202469825745,
    "resisc45": 0.6474603414535522,
    "eurosat": 0.732962965965271,
    "svhn": 0.7793484926223755,
    "gtsrb": 0.684956431388855,
    "mnist": 0.9606999754905701,
    "dtd": 0.47127658128738403,
    "average": 0.6754813715815544,
}

REFERENCE_FEATCAL_TA_ACCURACY = {
    "sun397": 0.700654923915863,
    "stanford-cars": 0.7251585721969604,
    "resisc45": 0.879365086555481,
    "eurosat": 0.9622222185134888,
    "svhn": 0.9505992531776428,
    "gtsrb": 0.9320665001869202,
    "mnist": 0.988099992275238,
    "dtd": 0.6994680762290955,
    "average": 0.8547043278813362,
}

