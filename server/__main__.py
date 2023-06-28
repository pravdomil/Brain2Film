import json
import os
import sys
from dataclasses import dataclass
from typing import List

# noinspection PyUnresolvedReferences
from google.colab import drive

input_dir = "/content/drive/MyDrive/AI Cut Ultra/input"
output_dir = "/content/drive/MyDrive/AI Cut Ultra/output"


# Types

@dataclass
class Initializing:
    _: None


@dataclass
class MountingDrive:
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
class Json:
    filename: None
    name: None
    clipStart: None
    clipDuration: None
    notes: None


# Functions

def __main__():
    try:
        state_step(Initializing(None))
    except KeyboardInterrupt:
        state_step(Exiting(None))


def state_step(a):
    if isinstance(a, Initializing):
        state_step(MountingDrive(None))

    elif isinstance(a, MountingDrive):
        print("Mounting Drive...")
        drive.mount("/content/drive")
        state_step(Ready(None))

    elif isinstance(a, Ready):
        print("Ready.")
        state_step(Checking(None))

    elif isinstance(a, Checking):
        print("Checking...")
        files = list_json_files(input_dir)
        json_strings = list(map(read_json, files))
        jsons = list(map(parse_json, json_strings))

    elif isinstance(a, Exiting):
        print("Quiting...")
        sys.exit()

    else:
        raise ValueError("Unknown variant.")


# Helpers
def parse_json(a) -> Json:
    if a[0] == "tdqt9rkbrsv7bf5bz16gy2p19" \
            and isinstance(a[1], str) \
            and isinstance(a[2], str) \
            and isinstance(a[3], str) \
            and isinstance(a[4], str) \
            and isinstance(a[5], str):
        return Json(a[1], a[2], a[3], a[4], a[5])
    else:
        raise ValueError("Cannot parse JSON.")


def list_json_files(directory: str) -> List[str]:
    acc = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            acc.append(os.path.join(directory, filename))
    return acc


def read_json(path: str) -> object:
    return json.load(open(path))


__main__()
