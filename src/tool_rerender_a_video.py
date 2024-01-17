import os

import PIL.Image
import PIL.ImageOps
import cv2
import diffusers
import moviepy.video.io.ffmpeg_writer
import torch

import config
import task


def main(arg: tuple[str, task.RerenderAVideo]):
    id_, a = arg

    input_filepath = os.path.join(config.input_dir, a.input_filename)
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(input_filepath)
    capture = cv2.VideoCapture(input_filepath)

    frame_indexes, fps = compute_frame_indexes_and_fps(a, capture.get(cv2.CAP_PROP_FPS))
    size = compute_size((capture.get(cv2.CAP_PROP_FRAME_WIDTH), capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    pipe = diffusers.StableDiffusionRerenderAVideoPipeline.from_pretrained("timbrooks/instruct-pix2pix", safety_checker=None).to(config.device)

    print("RerenderAVideo: \"" + a.prompt.replace("\n", "\\n") + "\", " + str(len(frame_indexes)) + " frames")

    writer = moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(
        os.path.join(config.output_dir, task.output_filename((id_, task.Task(a)), "mov")),
        size,
        fps,
        ffmpeg_params=["-crf", "15", "-metadata", "title=" + "\n".join(task.to_info(task.Task(a)))],
    )

    input_images: list[PIL.Image] = []
    for i in frame_indexes:
        image = capture_read_image(capture, i)
        resized_image = PIL.ImageOps.fit(image, size, method=PIL.Image.LANCZOS)
        input_images.append(resized_image)

    output_images = pipe(
        [a.prompt] * len(input_images),
        image=input_images,
        guidance_scale=(a.text_cfg if a.text_cfg is not None else 7),
        image_guidance_scale=(a.image_cfg if a.image_cfg is not None else 1),
        num_inference_steps=15,
        generator=torch.manual_seed(config.seed),
    ).images

    for image in output_images:
        writer.write_frame(image)

    capture.release()
    writer.close()


def compute_frame_indexes_and_fps(a: task.RerenderAVideo, fps: float) -> tuple[list[int], float]:
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


def capture_read_image(a, index: int) -> PIL.Image.Image:
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return PIL.Image.fromarray(rgb_image)
    else:
        raise Exception("Cannot read video frame.")
