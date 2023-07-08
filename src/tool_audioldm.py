import os

import diffusers
import torch

import config
import task
import utils


def main(a: task.AudioLDM):
    pipe = diffusers.AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float16
    ).to(config.device)

    torch.manual_seed(config.seed)
    audio = pipe(a.prompt, num_inference_steps=10, audio_length_in_s=a.duration).audios[0]

    utils.save_to_mp3(audio, os.path.join(config.output_dir, a.output_filename))
