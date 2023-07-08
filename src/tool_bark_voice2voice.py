import os

import bark
import bark.generation
import encodec.utils
import hubert.customtokenizer
import hubert.hubert_manager
import hubert.pre_kmeans_hubert
import torch
import torchaudio

import config
import task
import tool_bark_utils


def main(a: task.BarkVoice2Voice):
    print("Bark voice2voice: \"" + a.prompt.replace("\n", "\\n") + "\"")

    hubert_filepath = hubert.hubert_manager.ensure_hubert()
    tokenizer_filepath = hubert.hubert_manager.ensure_tokenizer()

    hubert_ = hubert.pre_kmeans_hubert.CustomHubert(hubert_filepath).to(config.device)
    tokenizer = hubert.customtokenizer.CustomTokenizer.load_from_checkpoint(tokenizer_filepath).to(config.device)
    codec = bark.generation.load_codec_model()
    torch.manual_seed(config.seed)

    wav = load_audio(codec, os.path.join(config.input_dir, a.input_filename))
    semantic_vectors = hubert_.forward(wav, input_sample_hz=codec.sample_rate)
    semantic_tokens = tokenizer.get_token(semantic_vectors)

    audio = bark.semantic_to_waveform(
        semantic_tokens.cpu().numpy(),
        history_prompt="v2/" + a.speaker[0] + "_speaker_" + str(a.speaker[1]),
    )

    tool_bark_utils.save_to_mp3(audio, os.path.join(config.output_dir, a.output_filename))


def load_audio(codec, a: str):
    # noinspection PyUnresolvedReferences
    wav, sample_rate = torchaudio.load(a)
    return encodec.utils.convert_audio(wav, sample_rate, codec.sample_rate, codec.channels).to(config.device)
