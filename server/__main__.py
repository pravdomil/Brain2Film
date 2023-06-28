import json
import os
import sys
from dataclasses import dataclass

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
def load_json_files(directory: str) -> object:
    json_data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path) as file:
                data = json.load(file)
                json_data.append(data)
    return json_data


__main__()
