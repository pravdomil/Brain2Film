from dataclasses import dataclass
from typing import Union


@dataclass
class InstructPix2Pix:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str
    fps: Union[int, None]
    text_cfg: Union[int, None]
    image_cfg: Union[int, None]


@dataclass
class Bark:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str


@dataclass
class AudioLDM:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str


@dataclass
class Audiocraft:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str


@dataclass
class Task:
    type: Union[InstructPix2Pix, Bark, AudioLDM, Audiocraft]


def encode(a: Task) -> object:
    if isinstance(a.type, InstructPix2Pix):
        return (
            "rvb3vnlcmjkhxdsf7yqr45m40", a.type.name, a.type.input_filename, a.type.output_filename, a.type.clip_start,
            a.type.clip_duration, a.type.prompt, a.type.fps, a.type.text_cfg, a.type.image_cfg)

    elif isinstance(a.type, Bark):
        return (
            "0f96skf4tvg74wjp6c9nn0sxk", a.type.name, a.type.input_filename, a.type.output_filename, a.type.clip_start,
            a.type.clip_duration, a.type.prompt)

    elif isinstance(a.type, AudioLDM):
        return (
            "y7l7hv8w5rq6nffn3tyyb_tfx", a.type.name, a.type.input_filename, a.type.output_filename, a.type.clip_start,
            a.type.clip_duration, a.type.prompt)

    elif isinstance(a.type, Audiocraft):
        return (
            "pnt5pvz6x6s3jwjxz2v8pcvy0", a.type.name, a.type.input_filename, a.type.output_filename, a.type.clip_start,
            a.type.clip_duration, a.type.prompt)

    else:
        raise ValueError("Unknown variant.")
