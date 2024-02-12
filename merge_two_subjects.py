import os
import argparse
import shutil

IMAGES_PATH = "./data/image"


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example of a ControlNet training script."
    )
    parser.add_argument(
        "--first",
        type=str,
        required=True,
        help="First directory to merge",
    )
    parser.add_argument(
        "--second",
        type=str,
        required=True,
        help="Second directory to merge",
    )
    return parser.parse_args(input_args)


def copy_with_new_name(to_dir, from_dir, old_name, new_name):
    # copy agnostic

    shutil.copy(
        os.path.join(from_dir, "agnostic", old_name),
        os.path.join(to_dir, "agnostic", new_name),
    )

    # copy clothes
    shutil.copy(
        os.path.join(from_dir, "clothes", old_name),
        os.path.join(to_dir, "clothes", new_name),
    )
    # copy head
    shutil.copy(
        os.path.join(from_dir, "head", old_name),
        os.path.join(to_dir, "head", new_name),
    )
    # copy mask
    shutil.copy(
        os.path.join(from_dir, "mask", old_name),
        os.path.join(to_dir, "mask", new_name),
    )
    # copy openpose image
    shutil.copy(
        os.path.join(from_dir, "openpose", old_name),
        os.path.join(to_dir, "openpose", new_name),
    )
    # copy openpose json
    shutil.copy(
        os.path.join(from_dir, "openpose", old_name.replace(".jpg", ".json")),
        os.path.join(to_dir, "openpose", new_name.replace(".jpg", ".json")),
    )
    # copy processed
    shutil.copy(
        os.path.join(from_dir, "processed", old_name),
        os.path.join(to_dir, "processed", new_name),
    )
    # copy subject
    shutil.copy(
        os.path.join(from_dir, "subject", old_name),
        os.path.join(to_dir, "subject", new_name),
    )


def merge_two_subjects(first_dir, second_dir):
    first = os.listdir(os.path.join(first_dir, "processed"))
    second = os.listdir(os.path.join(second_dir, "processed"))
    # files are with format: "processed/0.jpg", "processed/1.jpg", ...
    # find largest number in first_subjects
    max_first = max([int(subject.split(".")[0]) for subject in first])
    max_first += 1

    # rename second_subjects
    for i, subject in enumerate(second):
        new_name = f"{max_first + i}.jpg"
        old_name = subject
        copy_with_new_name(first_dir, second_dir, old_name, new_name)

    # rename second_dir to second_dir + "_skip_" to avoid processing it again
    os.rename(second_dir, second_dir + "_skip_")


if __name__ == "__main__":
    args = parse_args()
    first_dir = args.first
    second_dir = args.second
    if first_dir == second_dir:
        print("Directories are the same")
        exit()
    all_subjects = merge_two_subjects(first_dir, second_dir)
