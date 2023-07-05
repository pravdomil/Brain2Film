import os
import tempfile
from typing import Union

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
import moviepy.editor
import torch

import config
import task


def main(a: task.InstructPix2Pix):
    capture = cv2.VideoCapture(os.path.join(config.input_dir, a.input_filename))

    frame_indexes, final_fps = compute_frame_indexes(
        a, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), capture.get(cv2.CAP_PROP_FPS)
    )
    batches = group_by(frame_indexes, config.batch_size)

    temp_dir = tempfile.TemporaryDirectory()

    print("InstructPix2Pix: \"" + a.prompt.replace("\n", ", ") + "\", " + str(len(batches)) + " batches")

    frames = []
    first_run = True
    for batch in batches:
        input_images: list[tuple[str, PIL.Image]] = []
        for i in batch:
            image = capture_read_image(capture, i)
            if image is not None:
                input_images.append(("instruct_pix2pix " + str(i) + ".png", resize_image(image)))

        output_images = instruct_pix2pix2(
            [x[1] for x in input_images],
            a.prompt,
            15,
            123,
            a.text_cfg if a.text_cfg is not None else 7,
            a.image_cfg if a.image_cfg is not None else 1,
        )

        for (image_filename, _), image in zip(input_images, output_images):
            temp_filename = os.path.join(temp_dir.name, image_filename)
            image.save(temp_filename)
            frames.append(temp_filename)

        if first_run:
            images_to_video(a.output_filename, frames, final_fps)
            first_run = False

    capture.release()

    images_to_video(a.output_filename, frames, final_fps)


def compute_frame_indexes(a: task.InstructPix2Pix, frame_count: int, fps: float) -> tuple[list[int], float]:
    if a.fps is None:
        frame_skip = 1
        final_fps = fps
    else:
        frame_skip = max(1, round(fps / a.fps))
        final_fps = fps / frame_skip

    start_frame = int(a.clip_start[0] * fps + a.clip_start[1])
    end_frame = int(start_frame + a.clip_duration[0] * fps + a.clip_duration[1])
    frame_indexes = []
    i = 0
    while 1:
        frame_index = int(start_frame + i * frame_skip)
        if frame_index > frame_count - 1:
            break
        if frame_index > end_frame - 1:
            break
        frame_indexes.append(frame_index)
        i = i + 1

    return frame_indexes, final_fps


def images_to_video(output_filename: str, frames: list[str], fps: float):
    if frames:
        clip = moviepy.editor.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            os.path.join(config.output_dir, output_filename),
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
        steps: int,
        seed: int,
        text_cfg: float,
        image_cfg: float,
) -> list[PIL.Image.Image]:
    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    pipe.to("cuda")

    torch.use_deterministic_algorithms(True)

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