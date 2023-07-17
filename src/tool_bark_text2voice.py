import os

import bark
import torch

import config
import task
import utils


def main(arg: tuple[str, task.BarkText2Voice]):
    id_, a = arg

    print("Bark text2voice: \"" + a.prompt.replace("\n", "\\n") + "\"")

    torch.manual_seed(config.seed)
    audio = bark.generate_audio(a.prompt, history_prompt="v2/" + a.speaker[0] + "_speaker_" + str(a.speaker[1]))

    utils.save_to_mp3(
        audio,
        os.path.join(config.output_dir, task.output_filename((id_, task.Task(a)), "mp3")),
        bark.SAMPLE_RATE,
        "\n".join(task.to_info(task.Task(a))),
    )
