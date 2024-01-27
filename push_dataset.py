import datasets

DATASET_PATH = "data/pairs/"

dataset = datasets.load_from_disk(DATASET_PATH)

dataset.push_to_hub("andrei-ace/EdgeStyle", private=True)
