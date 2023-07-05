# noinspection PyPackageRequirements
import bark
import scipy

import task


def main(a: task.Bark):
    bark.preload_models()
    audio = bark.generate_audio(a.prompt)
    scipy.io.wavfile.write(a.output_filename, bark.SAMPLE_RATE, audio)

