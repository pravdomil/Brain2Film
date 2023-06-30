import json
import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from typing import Union, TextIO

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
# noinspection PyUnresolvedReferences
import google.colab
import moviepy.editor
import torch
from PIL.Image import Image

drive_dir = "/content/drive"
base_dir = os.path.join(drive_dir, "MyDrive/AI Cut Ultra")
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "output")
tasks_dir = os.path.join(base_dir, "tasks")
tasks_done_dir = os.path.join(base_dir, "tasks/done")
tasks_error_dir = os.path.join(base_dir, "tasks/error")


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
    fps: int


@dataclass
class Bark:
    prompt: str


@dataclass
class AudioLDM:
    prompt: str


@dataclass
class Audiocraft:
    prompt: str


@dataclass
class Task:
    name: str
    input_filename: str
    output_filename: str
    clip_start: str
    clip_duration: str
    type: Union[InstructPix2Pix, Bark, AudioLDM, Audiocraft]


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
            instructions = data[6].lower()

            if instructions.startswith("pix2pix"):
                first_line, rest_of_lines = (instructions + "\n").split("\n", 1)
                type_ = InstructPix2Pix(rest_of_lines.strip(), 1)
                return Task(data[1], data[2], data[3], data[4], data[5], type_)

            elif instructions.startswith("bark"):
                first_line, rest_of_lines = (instructions + "\n").split("\n", 1)
                type_ = Bark(rest_of_lines.strip())
                return Task(data[1], data[2], data[3], data[4], data[5], type_)

            elif instructions.startswith("audioldm"):
                first_line, rest_of_lines = (instructions + "\n").split("\n", 1)
                type_ = AudioLDM(rest_of_lines.strip())
                return Task(data[1], data[2], data[3], data[4], data[5], type_)

            elif instructions.startswith("audiocraft"):
                first_line, rest_of_lines = (instructions + "\n").split("\n", 1)
                type_ = Audiocraft(rest_of_lines.strip())
                return Task(data[1], data[2], data[3], data[4], data[5], type_)

            else:
                return None

        else:
            return None
    except:
        return None


def list_task_filenames() -> list[str]:
    acc = []
    for filename in os.listdir(tasks_dir):
        if filename.endswith(".json"):
            acc.append(filename)
    return acc


def do_task_from_filename(filename: str):
    task = parse_task_json(open(os.path.join(tasks_dir, filename)))

    if task is None:
        print("Cannot parse \"" + filename + "\".")
        move_task_to_error_folder(filename)
    else:
        do_task((filename, task))
        move_task_to_done_folder(filename)


def move_task_to_done_folder(filename: str):
    os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_done_dir, filename))


def move_task_to_error_folder(filename: str):
    os.rename(os.path.join(tasks_dir, filename), os.path.join(tasks_error_dir, filename))


def do_task(arg: tuple[str, Task]):
    filename, a = arg
    if isinstance(a.type, InstructPix2Pix):
        instruct_pix2pix(arg, a.type)

    elif isinstance(a.type, Bark):
        print("Bark!")

    elif isinstance(a.type, AudioLDM):
        print("AudioLDM!")

    elif isinstance(a.type, Audiocraft):
        print("Audiocraft!")

    else:
        raise ValueError("Unknown variant.")


# InstructPix2Pix

def instruct_pix2pix(arg: tuple[str, Task], b: InstructPix2Pix):
    filename, a = arg

    print("InstructPix2Pix: \"" + b.prompt.replace("\n", ", ") + "\"")

    # noinspection PyUnresolvedReferences
    capture = cv2.VideoCapture(os.path.join(input_dir, a.input_filename))

    # noinspection PyUnresolvedReferences
    frame_indexes, final_fps = compute_frame_indexes(b,
                                                     int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
                                                     capture.get(cv2.CAP_PROP_FPS)
                                                     )

    temp_dir = tempfile.TemporaryDirectory()

    frames = []
    first_run = True
    for group in group_by_eight(frame_indexes):
        batch: list[tuple[str, Image]] = []
        for i in group:
            image = capture_read_image(capture, i)
            if image is not None:
                batch.append(("instruct_pix2pix " + str(i) + ".png", resize_image(image)))

        images = instruct_pix2pix2([x[1] for x in batch], b.prompt)
        for (image_filename, _), image in zip(batch, images):
            temp_filename = os.path.join(temp_dir.name, image_filename)
            image.save(temp_filename)
            frames.append(temp_filename)

        if first_run:
            images_to_video(arg, frames, final_fps)
            first_run = False

    capture.release()

    images_to_video(arg, frames, final_fps)


def compute_frame_indexes(b: InstructPix2Pix, frame_count: int, fps: int) -> tuple[list[int], int]:
    frame_skip = max(1, round(fps / b.fps))
    final_fps = round(fps / frame_skip)

    frame_indexes = []
    for i in range(0, frame_count - 1):
        if i % frame_skip == 0:
            frame_indexes.append(i)

    return frame_indexes, final_fps


def images_to_video(arg: tuple[str, Task], frames: list[str], fps: int):
    filename, a = arg

    if frames:
        moviepy.editor.ImageSequenceClip(frames, fps=fps) \
            .write_videofile(os.path.join(output_dir, a.output_filename),
                             ffmpeg_params=["-crf", "15"],
                             logger=None,
                             )
        print("Video saved.")


def resize_image(a: PIL.Image.Image) -> PIL.Image.Image:
    width, height = a.size
    factor = 768 / max(width, height)
    scaled_width = int(width * factor)
    scaled_height = int(height * factor)
    return PIL.ImageOps.fit(a, (scaled_width, scaled_height), method=PIL.Image.LANCZOS)


def instruct_pix2pix2(
        images: list[PIL.Image.Image],
        prompt: str,
        steps: int = 15,
        seed: int = 123,
        text_cfg_scale: float = 7,
        image_cfg_scale: float = 1,
) -> list[PIL.Image.Image]:
    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to("cuda")

    output = pipe(
        [prompt] * len(images),
        image=images,
        guidance_scale=text_cfg_scale,
        image_guidance_scale=image_cfg_scale,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    )

    return output.images


def capture_read_image(a, index: int) -> Union[PIL.Image.Image, None]:
    # noinspection PyUnresolvedReferences
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        # noinspection PyUnresolvedReferences
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb_image)
    else:
        return None


def group_by_eight(a):
    return [a[i:i + 8] for i in range(0, len(a), 8)]


__main__()
