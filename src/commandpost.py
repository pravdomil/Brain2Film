import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import yaml

import task


def main():
    input_dir, output_dir, tasks_dir = check_drive()
    version, directories, filename, name, clip_start, clip_duration, instructions = json.loads(sys.argv[1])

    if version != "_dx2rgq3ln9kfsl_wdv9vzlng":
        raise "Version mismatch."

    filepath = find_file(directories, filename)
    filepath_extension = os.path.splitext(filepath)[1]

    task_id = str(round(time.time() * 1000))
    input_filename = compute_sha256(filepath) + filepath_extension
    output_filename = re.sub("^[:alnum:-_]", " ", name) + " " + task_id + filepath_extension

    # Copy file.
    shutil.copy(filepath, os.path.join(input_dir, input_filename))

    # Create task.
    task_ = parse_task(name, input_filename, output_filename, clip_start, clip_duration, instructions)
    with open(os.path.join(tasks_dir, task_id + ".json"), "w") as f:
        json.dump(task.encode(task_), f)


def check_drive():
    base_dir = os.path.join(os.path.expanduser("~"), "My Drive/AI Cut Ultra")
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    tasks_dir = os.path.join(base_dir, "tasks")

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


def parse_task(
        name: str, input_filename: str, output_filename: str,
        clip_start: str, clip_duration: str, instructions: str
) -> task.Task:
    if instructions.lower().startswith("pix2pix:"):
        c = yaml.safe_load(instructions)
        prompt = c["pix2pix"]
        fps = c["fps"] if "fps" in c else None
        text_cfg = c["text_cfg"] if "text_cfg" in c else None
        image_cfg = c["image_cfg"] if "image_cfg" in c else None

        if isinstance(prompt, str) \
                and ((fps is None) or isinstance(fps, int)) \
                and ((text_cfg is None) or isinstance(text_cfg, int)) \
                and ((image_cfg is None) or isinstance(image_cfg, int)):
            type_ = task.InstructPix2Pix(
                name, input_filename, output_filename, parse_time(clip_start), parse_time(clip_duration),
                prompt, fps, text_cfg, image_cfg
            )
            return task.Task(type_)
        else:
            raise "Unknown InstructPix2Pix instructions."

    elif instructions.lower().startswith("bark:"):
        c = yaml.safe_load(instructions)
        prompt = c["bark"]
        if "speaker" in c:
            speaker = (c["speaker"][:2], int(c["speaker"][2:]) - 1)
        else:
            speaker = ("en", 0)

        if isinstance(prompt, str):
            type_ = task.BarkText2Voice(
                name, input_filename, output_filename, parse_time(clip_start), parse_time(clip_duration),
                prompt, speaker
            )
            return task.Task(type_)
        else:
            raise "Unknown Bark instructions."

    elif instructions.lower().startswith("audioldm:"):
        c = yaml.safe_load(instructions)
        prompt = c["audioldm"]

        if isinstance(prompt, str):
            type_ = task.AudioLDM(
                name, input_filename, output_filename, parse_time(clip_start), parse_time(clip_duration),
                prompt,
            )
            return task.Task(type_)
        else:
            raise "Unknown AudioLDM instructions."

    elif instructions.lower().startswith("audiocraft:"):
        c = yaml.safe_load(instructions)
        prompt = c["audiocraft"]

        if isinstance(prompt, str):
            type_ = task.Audiocraft(
                name, input_filename, output_filename, parse_time(clip_start), parse_time(clip_duration),
                prompt,
            )
            return task.Task(type_)
        else:
            raise "Unknown Audiocraft instructions."

    else:
        raise "Unknown instructions."


def parse_time(a: str) -> tuple[int, int]:
    h, m, s, rest = map(int, a.split(":"))
    return int(h * 60 * 60 + m * 60 + s), rest


if __name__ == "__main__":
    try:
        main()
        subprocess.run(["say", "Done."])
    except Exception as e:
        traceback.print_exception(e)
        subprocess.run(["say", "Error."])
