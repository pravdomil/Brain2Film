import basicsr.archs.rrdbnet_arch
import realesrgan

import config
import task


def main(a: task.RealESRGAN):
    upsampler = realesrgan.RealESRGANer(
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        model=basicsr.archs.rrdbnet_arch.RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        scale=4,
        device=config.device,
    )
