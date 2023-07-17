import json
import re
from dataclasses import dataclass
from typing import Union

from beartype import beartype


@beartype
@dataclass
class InstructPix2Pix:
    name: str
    input_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str
    fps: Union[int, None]
    text_cfg: Union[int, None]
    image_cfg: Union[int, None]


@beartype
@dataclass
class RealESRGAN:
    name: str
    input_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]


@beartype
@dataclass
class BarkText2Voice:
    name: str
    prompt: str
    speaker: tuple[str, int]


@beartype
@dataclass
class BarkVoice2Voice:
    name: str
    input_filename: str
    speaker: tuple[str, int]


@beartype
@dataclass
class AudioLDM:
    name: str
    prompt: str
    duration: float


@beartype
@dataclass
class Audiocraft:
    name: str
    prompt: str
    duration: float


@beartype
@dataclass
class Task:
    type: Union[InstructPix2Pix, RealESRGAN, BarkText2Voice, BarkVoice2Voice, AudioLDM, Audiocraft]


def encode(a: Task) -> object:
    if isinstance(a.type, InstructPix2Pix):
        return (
            "rvb3vnlcmjkhxdsf7yqr45m40",
            a.type.name,
            a.type.input_filename,
            a.type.clip_start,
            a.type.clip_duration,
            a.type.prompt,
            a.type.fps,
            a.type.text_cfg,
            a.type.image_cfg,
        )

    elif isinstance(a.type, RealESRGAN):
        return (
            "v6yhq70lnl6k71kyfj870h1s4",
            a.type.name,
            a.type.input_filename,
            a.type.clip_start,
            a.type.clip_duration,
        )

    elif isinstance(a.type, BarkText2Voice):
        return (
            "0f96skf4tvg74wjp6c9nn0sxk",
            a.type.name,
            a.type.prompt,
            a.type.speaker,
        )

    elif isinstance(a.type, BarkVoice2Voice):
        return (
            "8tsbpcdxrrhwdkff3cbk7h8cn",
            a.type.name,
            a.type.input_filename,
            a.type.speaker,
        )

    elif isinstance(a.type, AudioLDM):
        return (
            "y7l7hv8w5rq6nffn3tyyb_tfx",
            a.type.name,
            a.type.prompt,
            a.type.duration,
        )

    elif isinstance(a.type, Audiocraft):
        return (
            "pnt5pvz6x6s3jwjxz2v8pcvy0",
            a.type.name,
            a.type.prompt,
            a.type.duration,
        )

    else:
        raise ValueError("Unknown variant.")


def decode(a: any) -> Task:
    b = json.load(a)

    if b[0] == "rvb3vnlcmjkhxdsf7yqr45m40":
        type_ = InstructPix2Pix(b[1], b[2], tuple(b[3]), tuple(b[4]), b[5], b[6], b[7], b[8])
        return Task(type_)

    elif b[0] == "v6yhq70lnl6k71kyfj870h1s4":
        type_ = RealESRGAN(b[1], b[2], tuple(b[3]), tuple(b[4]))
        return Task(type_)

    elif b[0] == "0f96skf4tvg74wjp6c9nn0sxk":
        type_ = BarkText2Voice(b[1], b[2], tuple(b[3]))
        return Task(type_)

    elif b[0] == "8tsbpcdxrrhwdkff3cbk7h8cn":
        type_ = BarkVoice2Voice(b[1], b[2], tuple(b[3]))
        return Task(type_)

    elif b[0] == "y7l7hv8w5rq6nffn3tyyb_tfx":
        type_ = AudioLDM(b[1], b[2], b[3])
        return Task(type_)

    elif b[0] == "pnt5pvz6x6s3jwjxz2v8pcvy0":
        type_ = Audiocraft(b[1], b[2], b[3])
        return Task(type_)

    else:
        raise ValueError("Unknown variant.")


def to_info(a: Task) -> list[str]:
    if isinstance(a.type, InstructPix2Pix):
        return [
            "InstructPix2Pix",
            str(a.type.prompt),
            str(a.type.fps),
            str(a.type.text_cfg),
            str(a.type.image_cfg),
        ]

    elif isinstance(a.type, RealESRGAN):
        return [
            "RealESRGAN",
        ]

    elif isinstance(a.type, BarkText2Voice):
        return [
            "Bark text2voice",
            str(a.type.prompt),
            str(a.type.speaker),
        ]

    elif isinstance(a.type, BarkVoice2Voice):
        return [
            "Bark voice2voice",
            str(a.type.speaker),
        ]

    elif isinstance(a.type, AudioLDM):
        return [
            "AudioLDM",
            str(a.type.prompt),
        ]

    elif isinstance(a.type, Audiocraft):
        return [
            "Audiocraft",
            str(a.type.prompt),
        ]

    else:
        raise ValueError("Unknown variant.")


def name(a: Task) -> str:
    if isinstance(a.type, InstructPix2Pix):
        return a.type.name
    elif isinstance(a.type, RealESRGAN):
        return a.type.name
    elif isinstance(a.type, BarkText2Voice):
        return a.type.name
    elif isinstance(a.type, BarkVoice2Voice):
        return a.type.name
    elif isinstance(a.type, AudioLDM):
        return a.type.name
    elif isinstance(a.type, Audiocraft):
        return a.type.name
    else:
        raise ValueError("Unknown variant.")


def output_filename(ext: str, arg: tuple[str, Task]) -> str:
    id_, a = arg
    return re.sub("[^0-9A-Za-z-_.]", " ", id_ + " " + name(a) + "." + ext)
