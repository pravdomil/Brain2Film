import os

import diffusers
import librosa
import numpy
import scipy.signal
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
    audio_highpass = apply_highpass_mix(audio_44khz)

    utils.save_to_mp3(audio_highpass, os.path.join(config.output_dir, a.output_filename))


def apply_highpass_mix(input: numpy.ndarray, highpass_vol=0.2, mix_vol=0.8, cutoff=11000, order=6):
    octave_up = librosa.effects.pitch_shift(input, sr=sample_rate, n_steps=12, res_type="soxr_vhq") * highpass_vol
    b, a = scipy.signal.butter(order, cutoff, fs=sample_rate, btype="highpass", analog=False)
    highpass = scipy.signal.lfilter(b, a, octave_up)
    return input * mix_vol + highpass
