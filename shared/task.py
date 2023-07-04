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
