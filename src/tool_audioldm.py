import os

import diffusers
import librosa
import numpy
import torch

import config
import task
import utils

sample_rate = 44100


def main(a: task.AudioLDM):
    print("AudioLDM: \"" + a.prompt.replace("\n", "\\n") + "\"")

    torch.manual_seed(config.seed)
    pipe = diffusers.AudioLDMPipeline.from_pretrained(
        "cvssp/audioldm-s-full-v2",
        torch_dtype=torch.float16
    ).to(config.device)

    audio = pipe(a.prompt, num_inference_steps=10, audio_length_in_s=a.duration).audios[0]
    audio_44khz = librosa.resample(audio, orig_sr=16000, target_sr=sample_rate)
    audio_enhanced = enhance_audio(audio_44khz)

    utils.save_to_mp3(audio_enhanced, os.path.join(config.output_dir, a.output_filename))


def enhance_audio(data: numpy.ndarray):
    octave_up = librosa.effects.pitch_shift(data, sr=sample_rate, n_steps=12, res_type="soxr_vhq")
    return data + octave_up
