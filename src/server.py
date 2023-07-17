import os
import time
import traceback
from dataclasses import dataclass

# noinspection PyUnresolvedReferences,PyPackageRequirements
import google.colab

import config
import task
import tool_audiocraft
import tool_audioldm
import tool_bark_text2voice
import tool_bark_voice2voice
import tool_instruct_pix2pix
import tool_realesrgan


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
class Error:
    exception: Exception


# Functions

def main():
    state = Initializing(None)

    while 1:
        try:
            state = step(state)

        except KeyboardInterrupt:
            break

        except Exception as e:
            state = Error(e)


def step(a):
    if isinstance(a, Initializing):
        print("Initializing...")

        if not os.path.exists(config.drive_dir):
            google.colab.drive.mount(config.drive_dir)

        if not os.path.exists(config.tasks_done_dir):
            os.makedirs(config.tasks_done_dir, exist_ok=True)

        if not os.path.exists(config.tasks_error_dir):
            os.makedirs(config.tasks_error_dir, exist_ok=True)

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

    elif isinstance(a, Error):
        traceback.print_exception(a.exception)
        print("Retrying in a second.")
        time.sleep(1)
        return Initializing(None)

    else:
        raise ValueError("Unknown variant.")


# Task

def list_task_filenames() -> list[str]:
    acc = []
    for filename in os.listdir(config.tasks_dir):
        if filename.endswith(".json"):
            acc.append(filename)
    return acc


def do_task_from_filename(filename: str):
    try:
        task_ = task.decode(open(os.path.join(config.tasks_dir, filename)))
        do_task((filename, task_))
        move_task_to_done_folder(filename)
        print("Task done \"" + filename + "\".")

    except Exception as e:
        traceback.print_exception(e)
        print("Error occurred during task \"" + filename + "\".")
        move_task_to_error_folder(filename)


def move_task_to_done_folder(filename: str):
    os.rename(os.path.join(config.tasks_dir, filename), os.path.join(config.tasks_done_dir, filename))


def move_task_to_error_folder(filename: str):
    os.rename(os.path.join(config.tasks_dir, filename), os.path.join(config.tasks_error_dir, filename))


def do_task(arg: tuple[str, task.Task]):
    id_, a = arg
    if isinstance(a.type, task.InstructPix2Pix):
        tool_instruct_pix2pix.main(a.type)

    elif isinstance(a.type, task.RealESRGAN):
        tool_realesrgan.main(a.type)

    elif isinstance(a.type, task.BarkText2Voice):
        tool_bark_text2voice.main(a.type)

    elif isinstance(a.type, task.BarkVoice2Voice):
        tool_bark_voice2voice.main(a.type)

    elif isinstance(a.type, task.AudioLDM):
        tool_audioldm.main(a.type)

    elif isinstance(a.type, task.Audiocraft):
        tool_audiocraft.main(a.type)

    else:
        raise ValueError("Unknown variant.")


if __name__ == "__main__":
    main()
