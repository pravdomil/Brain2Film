import os

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
import moviepy.video.io.ffmpeg_writer
import torch

import config
import task


def main(arg: tuple[str, task.InstructPix2Pix]):
    id_, a = arg

    input_filepath = os.path.join(config.input_dir, a.input_filename)
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(input_filepath)
    capture = cv2.VideoCapture(input_filepath)

    frame_indexes, fps = compute_frame_indexes_and_fps(a, capture.get(cv2.CAP_PROP_FPS))
    batches = group_by(frame_indexes, config.instruct_pix2pix_batch_size)
    size = compute_size((capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pipe = diffusers.StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", safety_checker=None).to(config.device)

    print("InstructPix2Pix: \"" + a.prompt.replace("\n", "\\n") + "\", " + str(len(batches)) + " batches")

    writer = moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(
        os.path.join(config.output_dir, task.output_filename((id_, task.Task(a)), "mov")),
        size,
        fps,
        ffmpeg_params=["-crf", "15", "-metadata", "title=" + "\n".join(task.to_info(task.Task(a)))],
    )

    for batch in batches:
        input_images: list[PIL.Image] = []
        for i in batch:
            image = capture_read_image(capture, i)
            resized_image = PIL.ImageOps.fit(image, size, method=PIL.Image.LANCZOS)
            input_images.append(resized_image)

        output_images = instruct_pix2pix2(
            pipe,
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
    factor = 512 / max(width, height)
    return int(width * factor), int(height * factor)


def instruct_pix2pix2(
        pipe,
        images: list[PIL.Image.Image],
        prompt: str,
        steps: int,
        seed: int,
        text_cfg: float,
        image_cfg: float,
) -> list[PIL.Image.Image]:
    output = pipe(
        [prompt] * len(images),
        image=images,
        guidance_scale=text_cfg,
        image_guidance_scale=image_cfg,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed),
    )

    return output.images


def capture_read_image(a, index: int) -> PIL.Image.Image:
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb_image)
    else:
        raise Exception("Cannot read video frame.")


def group_by(a: list, size: int):
    return [a[i:i + size] for i in range(0, len(a), size)]
