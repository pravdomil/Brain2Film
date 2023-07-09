import os
from typing import Union

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
import moviepy.video.io.ffmpeg_writer
import torch

import config
import task


def main(a: task.InstructPix2Pix):
    input_filepath = os.path.join(config.input_dir, a.input_filename)
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(input_filepath)
    capture = cv2.VideoCapture(input_filepath)

    frame_indexes, final_fps = compute_frame_indexes_and_fps(a, capture.get(cv2.CAP_PROP_FPS))
    batches = group_by(frame_indexes, config.batch_size)
    size = compute_size((capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print("InstructPix2Pix: \"" + a.prompt.replace("\n", "\\n") + "\", " + str(len(batches)) + " batches")

    writer = moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(
        os.path.join(config.output_dir, a.output_filename),
        size,
        final_fps,
        ffmpeg_params=["-crf", "15"],
    )

    for batch in batches:
        input_images: list[PIL.Image] = []
        for i in batch:
            image = capture_read_image(capture, i)
            if image is not None:
                resized_image = PIL.ImageOps.fit(image, size, method=PIL.Image.LANCZOS)
                input_images.append(resized_image)

        output_images = instruct_pix2pix2(
            input_images,
            a.prompt,
            15,
            config.seed,
            a.text_cfg if a.text_cfg is not None else 7,
            a.image_cfg if a.image_cfg is not None else 1,
        )

        for image in output_images:
            writer.write_frame(image)

    capture.release()
    writer.close()


def compute_frame_indexes_and_fps(a: task.InstructPix2Pix, fps: float) -> tuple[list[int], float]:
    if a.fps is None:
        frame_skip = 1
        final_fps = fps
    else:
        frame_skip = max(1, round(fps / a.fps))
        final_fps = fps / frame_skip

    start_frame = int(a.clip_start[0] * fps + a.clip_start[1])
    end_frame = int(start_frame + a.clip_duration[0] * fps + a.clip_duration[1])

    return list(range(start_frame, end_frame, frame_skip)), final_fps


def compute_size(size: tuple[float, float]) -> tuple[int, int]:
    width, height = size
    factor = 768 / max(width, height)
    return int(width * factor), int(height * factor)


def instruct_pix2pix2(
        images: list[PIL.Image.Image],
        prompt: str,
        steps: int,
        seed: int,
        text_cfg: float,
        image_cfg: float,
) -> list[PIL.Image.Image]:
    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(config.device)

    output = pipe(
        [prompt] * len(images),
        image=images,
        guidance_scale=text_cfg,
        image_guidance_scale=image_cfg,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    )

    return output.images


def capture_read_image(a, index: int) -> Union[PIL.Image.Image, None]:
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb_image)
    else:
        return None


def group_by(a: list, size: int):
    return [a[i:i + size] for i in range(0, len(a), size)]
