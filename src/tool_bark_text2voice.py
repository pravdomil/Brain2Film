import os

import bark
import torch

import config
import task
import utils


def main(a: task.BarkText2Voice):
    print("Bark text2voice: \"" + a.prompt.replace("\n", "\\n") + "\"")

    torch.manual_seed(config.seed)
    audio = bark.generate_audio(a.prompt, history_prompt="v2/" + a.speaker[0] + "_speaker_" + str(a.speaker[1]))

    utils.save_to_mp3(audio, os.path.join(config.output_dir, a.output_filename))
