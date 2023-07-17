import numpy
import pydub


def save_to_mp3(audio: numpy.ndarray, filename: str, sample_rate, title: str):
    normalized = numpy.int16(audio * 2 ** 15)
    segment = pydub.AudioSegment(normalized.tobytes(), sample_width=2, frame_rate=sample_rate, channels=1)
    segment.export(filename, format="mp3", bitrate="320k", tags={"title": title})
