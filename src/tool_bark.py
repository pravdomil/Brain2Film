import os

# noinspection PyPackageRequirements
import bark
import scipy

import config
import task


def main(a: task.Bark):
    print("Bark: \"" + a.prompt.replace("\n", "\\n") + "\"")

    audio = bark.generate_audio(a.prompt, history_prompt="v2/en_speaker_1")

    if a.output_filename.endswith(".wav"):
        output_filename = a.output_filename
    else:
        output_filename = a.output_filename + ".wav"

    scipy.io.wavfile.write(os.path.join(config.output_dir, output_filename), bark.SAMPLE_RATE, audio)
