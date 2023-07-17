import os

import diffusers
import librosa
import numpy
import torch

import config
import task
import utils

sample_rate = 44100


def main(arg: tuple[str, task.AudioLDM]):
    id_, a = arg

    print("AudioLDM: \"" + a.prompt.replace("\n", "\\n") + "\"")

    torch.manual_seed(config.seed)
    pipe = diffusers.AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float16
    ).to(config.device)

    audio = pipe(a.prompt, num_inference_steps=10, audio_length_in_s=a.duration).audios[0]
    audio_resampled = librosa.resample(audio, orig_sr=16000, target_sr=sample_rate)
    audio_enhanced = enhance_audio(audio_resampled)

    utils.save_to_mp3(
        audio_enhanced,
        os.path.join(config.output_dir, task.output_filename((id_, task.Task(a)), "mp3")),
        sample_rate,
        "\n".join(task.to_info(task.Task(a))),
    )


def enhance_audio(a: numpy.ndarray):
    octave_up = librosa.effects.pitch_shift(a, sr=sample_rate, n_steps=12, res_type="soxr_vhq")
    return a + octave_up
