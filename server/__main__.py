import json
import os
import sys
from dataclasses import dataclass
from typing import List

# noinspection PyUnresolvedReferences
from google.colab import drive


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
        print("Ready")

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


def load_json_files(directory: str) -> List[object]:
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path) as file:
                data = json.load(file)
                json_data.append(data)
    return json_data


__main__()
