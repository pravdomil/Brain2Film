# noinspection PyPackageRequirements
import bark
import scipy

import task


def main(a: task.Bark):
    print("Bark: \"" + a.prompt.replace("\n", ", ") + "\"")

    audio = bark.generate_audio(a.prompt)
    scipy.io.wavfile.write(a.output_filename, bark.SAMPLE_RATE, audio)
