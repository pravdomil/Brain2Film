import os.path
import shutil
import urllib.request

import huggingface_hub


def make_sure_hubert_installed(
        download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt',
        file_name: str = 'hubert.pt'):
    directory = os.path.join('data', 'models', 'hubert')
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, file_name)
    if not os.path.isfile(filepath):
        print("Downloading " + download_url)
        urllib.request.urlretrieve(download_url, filepath)

    return filepath


def make_sure_tokenizer_installed(
        model: str = 'quantifier_hubert_base_ls960_14.pth',
        repo: str = 'GitMylo/bark-voice-cloning',
        local_file: str = 'tokenizer.pth'):
    directory = os.path.join('data', 'models', 'hubert')
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    filepath = os.path.join(directory, local_file)
    if not os.path.isfile(filepath):
        print("Downloading " + model)
        huggingface_hub.hf_hub_download(repo, model, local_dir=directory, local_dir_use_symlinks=False)
        shutil.move(os.path.join(directory, model), filepath)

    return filepath
