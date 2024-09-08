from basicsr.archs.rrdbnet_arch import RRDBNet
from src.utils.Utils import RealESRGANer
from src import conf
import os
import cv2
import numpy as np

class GanModel:
    def __init__(self):
        self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        base = conf.BASE_URL
        model_name = "RealESRGAN_x4plus.pth"
        model_path = os.path.join(base, "weights", model_name)
        self.upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            dni_weight=None,
            model=self.model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
            gpu_id=None
        )

    def super_res(self, frame, outscale) -> np.ndarray:
        return self.upsampler.enhance(
            frame, outscale=outscale,
        )[0]


if __name__ == "__main__":
    gan_model = GanModel()
    pic = cv2.imread(os.path.join(conf.BASE_URL , "assets/comic.png"))
    super_res = gan_model.super_res(pic, 2)
    cv2.imshow("comic noob", pic)
    cv2.imshow("comic pro", super_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
