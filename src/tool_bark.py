import os

# noinspection PyPackageRequirements
import bark
import scipy

import config
import task


def main(a: task.Bark):
    print("Bark: \"" + a.prompt.replace("\n", ", ") + "\"")

    audio = bark.generate_audio(a.prompt)
    scipy.io.wavfile.write(os.path.join(config.output_dir, a.output_filename), bark.SAMPLE_RATE, audio)
