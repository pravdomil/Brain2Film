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
    fps: Union[float, None]
    text_cfg: Union[float, None]
    image_cfg: Union[float, None]


@beartype
@dataclass
class RerenderAVideo:
    name: str
    input_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str
    fps: Union[float, None]


@beartype
@dataclass
class FateZero:
    name: str
    input_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str
    fps: Union[float, None]
    cfg: Union[float, None]


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
    type: Union[InstructPix2Pix, RerenderAVideo, FateZero, RealESRGAN, BarkText2Voice, BarkVoice2Voice, AudioLDM, Audiocraft]


def encode(a: Task) -> object:
    if isinstance(a.type, InstructPix2Pix):
        return (
            "01HEGNWE6WT50DKEZ6E0YA10MS",
            a.type.name,
            a.type.input_filename,
            a.type.clip_start,
            a.type.clip_duration,
            a.type.prompt,
            a.type.fps,
            a.type.text_cfg,
            a.type.image_cfg,
        )

    elif isinstance(a.type, FateZero):
        return (
            "01HEGNWR360N0HRTDNG5AF458T",
            a.type.name,
            a.type.input_filename,
            a.type.clip_start,
            a.type.clip_duration,
            a.type.prompt,
            a.type.fps,
            a.type.cfg,
        )

    elif isinstance(a.type, RealESRGAN):
        return (
            "01HEGNWZ0SWBXY66TE32A1V5DA",
            a.type.name,
            a.type.input_filename,
            a.type.clip_start,
            a.type.clip_duration,
        )

    elif isinstance(a.type, BarkText2Voice):
        return (
            "01HEGNX5MTGFVYJKGJG4NKMJQV",
            a.type.name,
            a.type.prompt,
            a.type.speaker,
        )

    elif isinstance(a.type, BarkVoice2Voice):
        return (
            "01HEGNXCY0C12W70CJZH81VXT9",
            a.type.name,
            a.type.input_filename,
            a.type.speaker,
        )

    elif isinstance(a.type, AudioLDM):
        return (
            "01HEGNXK144HABSM294NW2QJN1",
            a.type.name,
            a.type.prompt,
            a.type.duration,
        )

    elif isinstance(a.type, Audiocraft):
        return (
            "01HEGNXRHQ09D9E6AAESK4GMDB",
            a.type.name,
            a.type.prompt,
            a.type.duration,
        )

    else:
        raise ValueError("Unknown variant.")


def decode(a: any) -> Task:
    b = json.load(a)

    if b[0] == "01HEGNWE6WT50DKEZ6E0YA10MS":
        type_ = InstructPix2Pix(b[1], b[2], tuple(b[3]), tuple(b[4]), b[5], maybe_map(float, b[6]),
                                maybe_map(float, b[7]), maybe_map(float, b[8]))
        return Task(type_)

    elif b[0] == "01HEGNWR360N0HRTDNG5AF458T":
        type_ = FateZero(b[1], b[2], tuple(b[3]), tuple(b[4]), b[5], maybe_map(float, b[6]),
                         maybe_map(float, b[7]))
        return Task(type_)

    elif b[0] == "01HEGNWZ0SWBXY66TE32A1V5DA":
        type_ = RealESRGAN(b[1], b[2], tuple(b[3]), tuple(b[4]))
        return Task(type_)

    elif b[0] == "01HEGNX5MTGFVYJKGJG4NKMJQV":
        type_ = BarkText2Voice(b[1], b[2], tuple(b[3]))
        return Task(type_)

    elif b[0] == "01HEGNXCY0C12W70CJZH81VXT9":
        type_ = BarkVoice2Voice(b[1], b[2], tuple(b[3]))
        return Task(type_)

    elif b[0] == "01HEGNXK144HABSM294NW2QJN1":
        type_ = AudioLDM(b[1], b[2], b[3])
        return Task(type_)

    elif b[0] == "01HEGNXRHQ09D9E6AAESK4GMDB":
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

    elif isinstance(a.type, RerenderAVideo):
        return [
            "RerenderAVideo",
            str(a.type.prompt),
            str(a.type.fps),
        ]

    elif isinstance(a.type, FateZero):
        return [
            "FateZero",
            str(a.type.prompt),
            str(a.type.fps),
            str(a.type.cfg),
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
    elif isinstance(a.type, FateZero):
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


def output_filename(arg: tuple[str, Task], ext: str) -> str:
    id_, a = arg
    return re.sub("[^0-9A-Za-z-_.]", " ", id_ + " " + name(a) + "." + ext)


def maybe_map(fn, a):
    if a is None:
        return a
    else:
        return fn(a)
