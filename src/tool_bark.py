import os

# noinspection PyPackageRequirements
import bark
import pydub

import config
import task


def main(a: task.Bark):
    print("Bark: \"" + a.prompt.replace("\n", "\\n") + "\"")

    audio = bark.generate_audio(a.prompt, history_prompt="v2/en_speaker_1")

    if a.output_filename.endswith(".mp3"):
        output_filename = a.output_filename
    else:
        output_filename = a.output_filename + ".mp3"

    segment = pydub.AudioSegment(audio.tobytes(), sample_width=2, frame_rate=bark.SAMPLE_RATE, channels=1)
    segment.export(os.path.join(config.output_dir, output_filename), format="mp3", bitrate="320k")
