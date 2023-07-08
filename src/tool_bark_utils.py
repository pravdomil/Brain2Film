import bark
import numpy
import pydub


def save_to_mp3(audio, filename: str) -> str:
    normalized = numpy.int16(audio * 2 ** 15)

    if filename.endswith(".mp3"):
        filename = filename
    else:
        filename = filename + ".mp3"

    segment = pydub.AudioSegment(normalized.tobytes(), sample_width=2, frame_rate=bark.SAMPLE_RATE, channels=1)
    segment.export(filename, format="mp3", bitrate="320k")

    return filename
