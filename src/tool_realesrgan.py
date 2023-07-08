import os
import tempfile
from typing import Union

import PIL.Image
import basicsr.archs.rrdbnet_arch
import cv2
import numpy
import realesrgan

import config
import task
import utils


def main(a: task.RealESRGAN):
    print("RealESRGAN: \"" + a.name + "\"")

    upsampler = realesrgan.RealESRGANer(
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=basicsr.archs.rrdbnet_arch.RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        scale=4,
        device=config.device,
    )

    capture = cv2.VideoCapture(os.path.join(config.input_dir, a.input_filename))

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_indexes = compute_frame_indexes(a, fps)
    temp_dir = tempfile.TemporaryDirectory()

    frames = []
    for i in frame_indexes:
        image = capture_read_image(capture, i)
        if image is not None:
            image = PIL.Image.fromarray(upsampler.enhance(image)[0])

            image_filename = "realesrgan " + str(i) + ".png"
            temp_filename = os.path.join(temp_dir.name, image_filename)
            image.save(temp_filename)
            frames.append(temp_filename)

    capture.release()

    utils.images_to_video(os.path.join(config.output_dir, a.output_filename), frames, fps)


def compute_frame_indexes(a: task.RealESRGAN, fps: float) -> list[int]:
    start_frame = int(a.clip_start[0] * fps + a.clip_start[1])
    end_frame = int(start_frame + a.clip_duration[0] * fps + a.clip_duration[1])
    return list(range(start_frame, end_frame))


def capture_read_image(a, index: int) -> Union[numpy.ndarray, None]:
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None
