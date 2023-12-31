import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import yaml

import task


def main():
    input_dir, output_dir, tasks_dir = check_drive()
    version, directories, filename, name, clip_start, clip_duration, notes = json.loads(sys.argv[1])

    if version != "01HEGNTBG1SFDYQN1WK843952H":
        raise Exception("Version mismatch.")

    filepath = find_file(directories, filename)
    filepath_extension = os.path.splitext(filepath)[1]

    task_id = str(round(time.time() * 1000))
    input_filename = compute_sha256(filepath) + filepath_extension

    # Copy file.
    shutil.copy(filepath, os.path.join(input_dir, input_filename))

    # Create task.
    task_ = parse_task(name, input_filename, clip_start, clip_duration, notes)
    with open(os.path.join(tasks_dir, task_id + ".json"), "w") as f:
        json.dump(task.encode(task_), f)


def check_drive():
    base_dir = os.path.join(os.path.expanduser("~"), "My Drive/Brain2Film")
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    tasks_dir = os.path.join(base_dir, "tasks")

    if not os.path.isdir(base_dir):
        raise Exception("Please create folder \"~/My Drive/Brain2Film\".")

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
        raise Exception("File not found.")

    if len(candidates) > 1:
        raise Exception("Multiple files found.")

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
        name: str, input_filename: str,
        clip_start: str, clip_duration: str, notes: str
) -> task.Task:
    if notes.lower().startswith("pix:"):
        c = yaml.safe_load(notes)
        prompt = c["pix"]
        fps = float(c["fps"]) if "fps" in c else None
        text_cfg = float(c["text_cfg"]) if "text_cfg" in c else None
        image_cfg = float(c["image_cfg"]) if "image_cfg" in c else None

        type_ = task.InstructPix2Pix(
            name,
            input_filename,
            parse_time(clip_start),
            parse_time(clip_duration),
            prompt,
            fps,
            text_cfg,
            image_cfg,
        )
        return task.Task(type_)

    elif notes.lower().startswith("fate:"):
        c = yaml.safe_load(notes)
        prompt = c["fate"]
        fps = float(c["fps"]) if "fps" in c else None
        cfg = float(c["cfg"]) if "cfg" in c else None

        type_ = task.FateZero(
            name,
            input_filename,
            parse_time(clip_start),
            parse_time(clip_duration),
            prompt,
            fps,
            cfg,
        )
        return task.Task(type_)

    elif notes.lower().startswith("up:"):
        type_ = task.RealESRGAN(
            name,
            input_filename,
            parse_time(clip_start),
            parse_time(clip_duration),
        )
        return task.Task(type_)

    elif notes.lower().startswith("bark:"):
        c = yaml.safe_load(notes)
        prompt = c["bark"]

        if "speaker" in c:
            speaker = (c["speaker"][:2], int(c["speaker"][2:]) - 1)
        else:
            speaker = ("en", 0)

        type_ = task.BarkText2Voice(
            name,
            prompt,
            speaker,
        )
        return task.Task(type_)

    elif notes.lower().startswith("voice:"):
        c = yaml.safe_load(notes)

        if "speaker" in c:
            speaker = (c["speaker"][:2], int(c["speaker"][2:]) - 1)
        else:
            speaker = ("en", 0)

        type_ = task.BarkVoice2Voice(
            name,
            input_filename,
            speaker,
        )
        return task.Task(type_)

    elif notes.lower().startswith("ldm:"):
        c = yaml.safe_load(notes)
        prompt = c["ldm"]
        duration = parse_time(clip_duration)
        duration = duration[0] + (duration[1] / 100)

        type_ = task.AudioLDM(
            name,
            prompt,
            duration,
        )
        return task.Task(type_)

    elif notes.lower().startswith("craft:"):
        c = yaml.safe_load(notes)
        prompt = c["craft"]
        duration = parse_time(clip_duration)
        duration = duration[0] + (duration[1] / 100)

        type_ = task.Audiocraft(
            name,
            prompt,
            duration,
        )
        return task.Task(type_)

    else:
        raise Exception("Notes must start with: pix, fate, up, bark, voice, ldm, craft.")


def parse_time(a: str) -> tuple[int, int]:
    h, m, s, rest = map(int, a.split(":"))
    return int(h * 60 * 60 + m * 60 + s), rest


if __name__ == "__main__":
    try:
        main()
        subprocess.run(["say", "Done."])
    except Exception as e:
        traceback.print_exception(e)
        subprocess.run(["say", str(e)])
