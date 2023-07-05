import os
import sys
import time
import traceback
from dataclasses import dataclass

# noinspection PyUnresolvedReferences,PyPackageRequirements
import google.colab

import globals
import instruct_pix2pix
import task


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


# Functions

def main():
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

        if not os.path.exists(globals.drive_dir):
            google.colab.drive.mount(globals.drive_dir)

        if not os.path.exists(globals.tasks_done_dir):
            os.makedirs(globals.tasks_done_dir, exist_ok=True)

        if not os.path.exists(globals.tasks_error_dir):
            os.makedirs(globals.tasks_error_dir, exist_ok=True)

        print("Done.")

        return Ready(None)

    elif isinstance(a, Ready):
        return Checking(None)

    elif isinstance(a, Checking):
        files = list_task_filenames()
        files.sort()

        if files:
            do_task_from_filename(files[0])
        else:
            time.sleep(1)

        return Ready(None)

    elif isinstance(a, Exiting):
        print("Quiting...")
        sys.exit()

    elif isinstance(a, Error):
        traceback.print_exception(a.exception)
        print("Retrying in a second.")
        time.sleep(1)
        return Ready(None)

    else:
        raise ValueError("Unknown variant.")


# Task

def list_task_filenames() -> list[str]:
    acc = []
    for filename in os.listdir(globals.tasks_dir):
        if filename.endswith(".json"):
            acc.append(filename)
    return acc


def do_task_from_filename(filename: str):
    task_ = task.decode(open(os.path.join(globals.tasks_dir, filename)))

    if task_ is None:
        print("Cannot parse \"" + filename + "\".")
        move_task_to_error_folder(filename)
    else:
        do_task((filename, task_))
        move_task_to_done_folder(filename)


def move_task_to_done_folder(filename: str):
    os.rename(os.path.join(globals.tasks_dir, filename), os.path.join(globals.tasks_done_dir, filename))


def move_task_to_error_folder(filename: str):
    os.rename(os.path.join(globals.tasks_dir, filename), os.path.join(globals.tasks_error_dir, filename))


def do_task(arg: tuple[str, task.Task]):
    filename, a = arg
    if isinstance(a.type, task.InstructPix2Pix):
        instruct_pix2pix.main(a.type)

    elif isinstance(a.type, task.Bark):
        bark(a.type)

    elif isinstance(a.type, task.AudioLDM):
        audioldm(a.type)

    elif isinstance(a.type, task.Audiocraft):
        audiocraft(a.type)

    else:
        raise ValueError("Unknown variant.")


# Bark

def bark(a: task.Bark):
    print("Bark!")


# AudioLDM

def audioldm(a: task.AudioLDM):
    print("AudioLDM!")


# Audiocraft

def audiocraft(a: task.Audiocraft):
    print("Audiocraft!")


if __name__ == "__main__":
    main()
