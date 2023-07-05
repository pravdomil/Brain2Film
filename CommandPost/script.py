import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time


def main():
    input_dir, output_dir, tasks_dir = check_drive()
    version, directories, f, name, clip_start, clip_duration, instructions = json.loads(sys.argv[1])

    if version != "_dx2rgq3ln9kfsl_wdv9vzlng":
        raise "Version mismatch."

    filepath = find_file(directories, f)
    filepath_extension = os.path.splitext(filepath)[1]

    task_id = str(round(time.time() * 1000))
    input_filename = compute_sha256(filepath) + filepath_extension
    output_filename = re.sub("^[:alnum:-_]", " ", name) + task_id + filepath_extension

    # Copy file.
    shutil.copy(filepath, os.path.join(input_dir, input_filename))

    # Create task.
    task = (
        "tdqt9rkbrsv7bf5bz16gy2p19",
        name,
        input_filename,
        output_filename,
        clip_start,
        clip_duration,
        instructions,
    )
    with open(os.path.join(tasks_dir, task_id + ".json"), 'w') as f:
        json.dump(task, f)

    subprocess.run(["say", "Done."])


def check_drive():
    base_dir = os.path.join(os.path.expanduser("~"), "My Drive/AI Cut Ultra")
    input_dir = os.path.join(base_dir + "input")
    output_dir = os.path.join(base_dir + "output")
    tasks_dir = os.path.join(base_dir + "tasks")

    if not os.path.isdir(base_dir):
        raise "Please create folder \"~/My Drive/AI Cut Ultra\"."

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)

    return input_dir, output_dir, tasks_dir


def compute_sha256(filepath: str) -> str:
    with open(filepath, "rb") as f:
        # noinspection PyTypeChecker
        return hashlib.file_digest(f, "sha256").hexdigest()


def find_file(directories: list[str], filename: str) -> str:
    candidates = find_files_by_name_in_directories(directories, filename)

    if not candidates:
        raise "File not found."

    if len(candidates) > 1:
        raise "Multiple files found."

    return candidates[0]


def find_files_by_name_in_directories(directories: list[str], filename: str) -> list[str]:
    acc = []
    for a in directories:
        acc.extend(find_files_by_name_in_directory(a, filename))
    return acc


def find_files_by_name_in_directory(directory: str, filename: str) -> list[str]:
    acc = []
    for root, _, files in os.walk(directory):
        for a in files:
            if a == filename:
                acc.append(os.path.join(root, a))
    return acc


if __name__ == "__main__":
    main()
