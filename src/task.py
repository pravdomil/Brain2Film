import json
from dataclasses import dataclass
from typing import Union

from beartype import beartype


@beartype
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


@beartype
@dataclass
class BarkText2Voice:
    name: str
    output_filename: str

    prompt: str
    speaker: tuple[str, int]


@beartype
@dataclass
class BarkVoice2Voice:
    name: str
    input_filename: str
    output_filename: str

    speaker: tuple[str, int]


@beartype
@dataclass
class AudioLDM:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str


@beartype
@dataclass
class Audiocraft:
    name: str
    input_filename: str
    output_filename: str
    clip_start: tuple[int, int]
    clip_duration: tuple[int, int]

    prompt: str


@beartype
@dataclass
class Task:
    type: Union[InstructPix2Pix, BarkText2Voice, BarkVoice2Voice, AudioLDM, Audiocraft]


def encode(a: Task) -> object:
    if isinstance(a.type, InstructPix2Pix):
        return (
            "rvb3vnlcmjkhxdsf7yqr45m40", a.type.name, a.type.input_filename, a.type.output_filename, a.type.clip_start,
            a.type.clip_duration, a.type.prompt, a.type.fps, a.type.text_cfg, a.type.image_cfg)

    elif isinstance(a.type, BarkText2Voice):
        return (
            "0f96skf4tvg74wjp6c9nn0sxk", a.type.name, a.type.output_filename, a.type.prompt, a.type.speaker)

    elif isinstance(a.type, BarkVoice2Voice):
        return (
            "8tsbpcdxrrhwdkff3cbk7h8cn", a.type.name, a.type.input_filename, a.type.output_filename, a.type.speaker)

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


def decode(a: any) -> Task:
    b = json.load(a)

    if b[0] == "rvb3vnlcmjkhxdsf7yqr45m40":
        type_ = InstructPix2Pix(b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9])
        return Task(type_)

    elif b[0] == "0f96skf4tvg74wjp6c9nn0sxk":
        type_ = BarkText2Voice(b[1], b[2], b[3], b[4])
        return Task(type_)

    elif b[0] == "8tsbpcdxrrhwdkff3cbk7h8cn":
        type_ = BarkVoice2Voice(b[1], b[2], b[3], b[4])
        return Task(type_)

    elif b[0] == "y7l7hv8w5rq6nffn3tyyb_tfx":
        type_ = AudioLDM(b[1], b[2], b[3], b[4], b[5], b[6])
        return Task(type_)

    elif b[0] == "pnt5pvz6x6s3jwjxz2v8pcvy0":
        type_ = Audiocraft(b[1], b[2], b[3], b[4], b[5], b[6])
        return Task(type_)

    else:
        raise ValueError("Unknown variant.")
