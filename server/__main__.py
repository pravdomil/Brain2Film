import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Union, TextIO

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
# noinspection PyUnresolvedReferences
import google.colab
import moviepy.editor
import torch

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
class InstructPix2Pix:
    prompt: str


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

        if not os.path.exists(drive_dir):
            google.colab.drive.mount(drive_dir)

        if not os.path.exists(tasks_done_dir):
            os.makedirs(tasks_done_dir)

        if not os.path.exists(tasks_error_dir):
            os.makedirs(tasks_error_dir)

        return Ready(None)

    elif isinstance(a, Ready):
        return Checking(None)

    elif isinstance(a, Checking):
        print("Checking...")
        files = list_task_filenames()
        files.sort()

        if files:
            do_task(files[0])
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


# Helpers

def parse_task_json(a: TextIO) -> Union[None, Task]:
    try:
        data = json.load(a)
        if data[0] == "tdqt9rkbrsv7bf5bz16gy2p19" \
                and isinstance(data[1], str) \
                and isinstance(data[2], str) \
                and isinstance(data[3], str) \
                and isinstance(data[4], str) \
                and isinstance(data[5], str) \
                and isinstance(data[6], str):
            return Task(data[1], data[2], data[3], data[4], data[5], data[6])
        else:
            return None
    except:
        return None


def list_task_filenames() -> List[str]:
    acc = []
    for filename in os.listdir(tasks_dir):
        if filename.endswith(".json"):
            acc.append(filename)
    return acc


# Task

def do_task(filename: str):
    task = parse_task_json(open(os.path.join(tasks_dir, filename)))

    if task is None:
        print("Cannot parse \"" + filename + "\".")
        os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_error_dir, filename))
    else:
        do_task2((filename, task))


def do_task2(arg: Tuple[str, Task]):
    filename, a = arg
    if a.instructions.lower().startswith("pix2pix"):
        instruct_pix2pix(arg)

    elif a.instructions.lower().startswith("bark"):
        print("Bark!")
        time.sleep(1)

    elif a.instructions.lower().startswith("audioldm"):
        print("AudioLDM!")
        time.sleep(1)

    elif a.instructions.lower().startswith("audiocraft"):
        print("Audiocraft!")
        time.sleep(1)

    else:
        print("Unknown instructions. It should be either pix2pix, bark, audioldm or audiocraft.")
        os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_error_dir, filename))


def instruct_pix2pix(arg: Tuple[str, Task]):
    def images_to_video():
        if frames:
            moviepy.editor.ImageSequenceClip(frames, fps=fps) \
                .write_videofile(os.path.join(output_dir, a.output_filename),
                                 fps=fps,
                                 ffmpeg_params=["-crf", "15"],
                                 logger=None,
                                 )
            print("Video saved.")

    filename, a = arg

    first_line, rest_of_lines = (a.instructions + "\n").split("\n", 1)
    data = InstructPix2Pix(rest_of_lines.strip())

    print("InstructPix2Pix: \"" + data.prompt.replace("\n", ", ") + "\"")

    temp_dir = tempfile.TemporaryDirectory()

    # noinspection PyUnresolvedReferences
    capture = cv2.VideoCapture(os.path.join(input_dir, a.input_filename))
    # noinspection PyUnresolvedReferences
    fps = capture.get(cv2.CAP_PROP_FPS)

    frames = []
    while 1:
        image = capture_read_image(capture)
        if image is None:
            break

        else:
            if len(frames) == 1 or len(frames) % 10 == 0:
                images_to_video()

            temp_filename = os.path.join(temp_dir.name, "instruct_pix2pix " + str(len(frames)) + ".png")
            output_image = instruct_pix2pix2(image, data.prompt)
            output_image.save(temp_filename)
            frames.append(temp_filename)

    capture.release()

    images_to_video()


def instruct_pix2pix2(
        image: PIL.Image.Image,
        prompt: str,
        steps: int = 15,
        seed: int = 123,
        text_cfg_scale: float = 7,
        image_cfg_scale: float = 1.5,
) -> PIL.Image.Image:
    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to("cuda")

    width, height = image.size
    factor = 768 / max(width, height)
    scaled_width = int(width * factor)
    scaled_height = int(height * factor)

    resized_image = PIL.ImageOps.fit(image, (scaled_width, scaled_height), method=PIL.Image.LANCZOS)

    output = pipe(
        prompt,
        image=resized_image,
        guidance_scale=text_cfg_scale,
        image_guidance_scale=image_cfg_scale,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    )

    return output.images[0]


def capture_read_image(a) -> Union[PIL.Image.Image, None]:
    retval, image = a.read()
    if retval:
        # noinspection PyUnresolvedReferences
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb_image)
    else:
        return None


__main__()
