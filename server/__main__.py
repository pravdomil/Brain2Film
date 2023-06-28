import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Union

# noinspection PyUnresolvedReferences
from google.colab import drive

drive_dir = "/content/drive"
input_dir = "/content/drive/MyDrive/AI Cut Ultra/input"
output_dir = "/content/drive/MyDrive/AI Cut Ultra/output"
tasks_dir = "/content/drive/MyDrive/AI Cut Ultra/tasks"
tasks_done_dir = "/content/drive/MyDrive/AI Cut Ultra/tasks/done"
tasks_error_dir = "/content/drive/MyDrive/AI Cut Ultra/tasks/error"


# Types

@dataclass
class Initializing:
    _: None


@dataclass
class Ready:
    _: None


@dataclass
class Checking:
    _: None


@dataclass
class Exiting:
    _: None


@dataclass
class Error:
    exception: Exception


@dataclass
class Task:
    name: str
    input_filename: str
    output_filename: str
    clip_start: str
    clip_duration: str
    instructions: str


# Functions

def __main__():
    state = Initializing(None)

    while 1:
        try:
            state = step(state)

        except KeyboardInterrupt:
            state = Exiting(None)

        except Exception as e:
            state = Error(e)


def step(a):
    if isinstance(a, Initializing):
        print("Initializing...")

        if not os.path.exists(tasks_done_dir):
            os.makedirs(tasks_done_dir)

        if not os.path.exists(tasks_error_dir):
            os.makedirs(tasks_error_dir)

        if not os.path.exists(drive_dir):
            drive.mount(drive_dir)

        return Ready(None)

    elif isinstance(a, Ready):
        return Checking(None)

    elif isinstance(a, Checking):
        print("Checking...")
        files = list_task_filenames()
        if files:
            do_task(files[0])
        else:
            time.sleep(1)

        return Ready(None)

    elif isinstance(a, Exiting):
        print("Quiting...")
        sys.exit()

    elif isinstance(a, Error):
        print(a.exception)
        print("Retrying in a second.")
        time.sleep(1)
        return Ready(None)

    else:
        raise ValueError("Unknown variant.")


# Helpers

def parse_task_json(a: any) -> Union[None, Task]:
    if a[0] == "tdqt9rkbrsv7bf5bz16gy2p19" \
            and isinstance(a[1], str) \
            and isinstance(a[2], str) \
            and isinstance(a[3], str) \
            and isinstance(a[4], str) \
            and isinstance(a[5], str) \
            and isinstance(a[6], str):
        return Task(a[1], a[2], a[3], a[4], a[5], a[6])
    else:
        return None


def list_task_filenames() -> List[str]:
    acc = []
    for filename in os.listdir(tasks_dir):
        if filename.endswith(".json"):
            acc.append(filename)
    acc.sort()
    return acc


# Task

def do_task(filename: str):
    data = json.load(open(os.path.join(tasks_dir, filename)))
    task = parse_task_json(data)

    if task is None:
        print("Cannot parse \"" + filename + "\".")
        os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_error_dir, filename))
    else:
        do_task2((filename, task))


def do_task2(arg: Tuple[str, Task]):
    filename, a = arg
    if a.instructions.lower().startswith("pix2pix"):
        print("Doing instruct InstructPix2Pix")
        time.sleep(1)

    elif a.instructions.lower().startswith("bark"):
        print("Doing Bark")
        time.sleep(1)

    elif a.instructions.lower().startswith("audioldm"):
        print("Doing AudioLDM")
        time.sleep(1)

    elif a.instructions.lower().startswith("audiocraft"):
        print("Doing Audiocraft")
        time.sleep(1)

    else:
        print("Unknown instructions. It should be either pix2pix, bark, audioldm or audiocraft.")
        os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_error_dir, filename))


__main__()
