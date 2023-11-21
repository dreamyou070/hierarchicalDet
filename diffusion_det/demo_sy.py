import glob
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer

# constants
WINDOW_NAME = "COCO detections"


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


import multiprocessing as mp
import argparse
from detectron2.utils.logger import setup_logger

def main(args) :

    print(f'\n step 1. set logger')
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    print(f'\n step 2. get argument')
    print(f' (1)  basic config')
    cfg = get_cfg()
    print(f' (2) model config')
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    print(f' (3) inference config')
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    print(f' (4) add score info')
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    print("Configuration: " + cfg)



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file",
                        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
                        metavar="FILE", help="path to config file",)
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+",
                        help="A list of space separated input images; or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",
                        help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.",)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    main(args)