import os

import torch
import transformers

import config
import task
import utils


def main(a: task.Audiocraft):
    print("Audiocraft: \"" + a.prompt.replace("\n", "\\n") + "\"")

    torch.manual_seed(config.seed)
    processor = transformers.AutoProcessor.from_pretrained("facebook/musicgen-large")
    model = transformers.MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

    inputs = processor(text=[a.prompt], padding=True, return_tensors="pt")
    audio = model.generate(**inputs, max_new_tokens=256)[0]

    utils.save_to_mp3(
        audio.numpy(),
        os.path.join(config.output_dir, a.output_filename),
        model.config.audio_encoder.sampling_rate,
        ", ".join(task.to_info(task.Task(a))),
    )
