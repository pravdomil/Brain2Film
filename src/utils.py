import moviepy.editor
import numpy
import pydub


def save_to_mp3(audio: numpy.ndarray, filename: str, sample_rate=44100) -> str:
    if filename.endswith(".mp3"):
        filename = filename
    else:
        filename = filename + ".mp3"

    normalized = numpy.int16(audio * 2 ** 15)
    segment = pydub.AudioSegment(normalized.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1)
    segment.export(filename, format="mp3", bitrate="320k")

    return filename


def images_to_video(filename: str, frames: list[str], fps: float):
    if frames:
        clip = moviepy.editor.ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(
            filename,
            ffmpeg_params=["-crf", "15"],
            logger=None,
        )
        print("Video saved.")
