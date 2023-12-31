import os

import PIL.Image
import basicsr.archs.rrdbnet_arch
import cv2
import moviepy.editor
import moviepy.video.io.ffmpeg_writer
import numpy
import realesrgan

import config
import task

scale = 4


def main(arg: tuple[str, task.RealESRGAN]):
    id_, a = arg

    print("RealESRGAN: \"" + a.name + "\"")

    input_filepath = os.path.join(config.input_dir, a.input_filename)
    if not os.path.isfile(input_filepath):
        raise FileNotFoundError(input_filepath)
    capture = cv2.VideoCapture(input_filepath)

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_indexes = compute_frame_indexes(a, fps)
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale))

    writer = moviepy.video.io.ffmpeg_writer.FFMPEG_VideoWriter(
        os.path.join(config.output_dir, task.output_filename((id_, task.Task(a)), "mov")),
        size,
        fps,
        ffmpeg_params=["-crf", "15", "-metadata", "title=" + "\n".join(task.to_info(task.Task(a)))],
    )
    upsampler = realesrgan.RealESRGANer(
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=basicsr.archs.rrdbnet_arch.RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale),
        scale=scale,
        device=config.device,
    )

    for i, frame_index in enumerate(frame_indexes):
        if i % 10 == 0:
            print(str(round(i / len(frame_indexes) * 100)) + "%")
        image = capture_read_image(capture, frame_index)
        enhanced_image = PIL.Image.fromarray(upsampler.enhance(image)[0])
        writer.write_frame(enhanced_image)

    capture.release()
    writer.close()


def compute_frame_indexes(a: task.RealESRGAN, fps: float) -> list[int]:
    start_frame = int(a.clip_start[0] * fps + a.clip_start[1])
    end_frame = int(start_frame + a.clip_duration[0] * fps + a.clip_duration[1])
    return list(range(start_frame, end_frame))


def capture_read_image(a, index: int) -> numpy.ndarray:
    a.set(cv2.CAP_PROP_POS_FRAMES, index)
    retval, image = a.read()
    if retval:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise Exception("Cannot read video frame.")
