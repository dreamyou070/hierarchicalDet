import numpy as np
import os
import tempfile
import pickle
from diffusiondet.predictor import AsyncPredictor
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs
from collections import deque
import cv2
import tqdm
import torch
import multiprocessing as mp
import argparse
from detectron2.utils.logger import setup_logger
from detectron2 import get_cfg, MetadataCatalog, VideoVisualizer,build_model,DetectionCheckpointer, PathManager
from detectron2.utils.visualizer import ColorMode, Visualizer
import detectron2.data.transforms as T
from fvcore.common.checkpoint import Checkpointer
import detectron2.utils.comm as comm
from torch.nn.parallel import DistributedDataParallel
from detectron2.data.detection_utils import read_image
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

class DetectionCheckpointer(Checkpointer):
    """
    Same as :class:`Checkpointer`, but is able to:
    1. handle models in detectron & detectron2 model zoo, and apply conversions for legacy models.
    2. correctly load checkpoints that are only available on the master worker
    """
    def __init__(self, model,
                 save_dir="", *, save_to_disk=None, **checkpointables):
        is_main_process = comm.is_main_process()
        super().__init__(model,
                         save_dir,
                         save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
                         **checkpointables,)
        self.path_manager = PathManager

    def load(self, path, *args, **kwargs):
        need_sync = False
        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                print(f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume.")
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable
        ret = super().load(path, *args, **kwargs)

        if need_sync:
            print("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        return ret

    def _load_file(self, filename):
        if filename.endswith(".pkl"):
            with PathManager.open(filename, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            if "model" in data and "__author__" in data:
                # file is in Detectron2 model zoo format
                self.logger.info("Reading a file from '{}'".format(data["__author__"]))
                return data
            else:
                # assume file is from Caffe2 / Detectron1 model zoo
                if "blobs" in data:
                    # Detection models have "blobs", but ImageNet models don't
                    data = data["blobs"]
                data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
                return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}
        elif filename.endswith(".pyth"):
            # assume file is from pycls; no one else seems to use the ".pyth" extension
            with PathManager.open(filename, "rb") as f:
                data = torch.load(f)
            assert (
                "model_state" in data
            ), f"Cannot load .pyth file {filename}; pycls checkpoints must contain 'model_state'."
            model_state = {
                k: v
                for k, v in data["model_state"].items()
                if not k.endswith("num_batches_tracked")
            }
            return {"model": model_state, "__author__": "pycls", "matching_heuristics": True}

        loaded = super()._load_file(filename)  # load native pth checkpoint
        if "model" not in loaded:
            loaded = {"model": loaded}
        loaded["matching_heuristics"] = True
        return loaded

    def _load_model(self, checkpoint):
        if checkpoint.get("matching_heuristics", False):
            self._convert_ndarray_to_tensor(checkpoint["model"])
            # convert weights by name-matching heuristics
            checkpoint["model"] = align_and_update_state_dicts(
                self.model.state_dict(),
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
        # for non-caffe2 models, use standard ways to load it
        incompatible = super()._load_model(checkpoint)

        model_buffers = dict(self.model.named_buffers(recurse=False))
        for k in ["pixel_mean", "pixel_std"]:
            # Ignore missing key message about pixel_mean/std.
            # Though they may be missing in old checkpoints, they will be correctly
            # initialized from config anyway.
            if k in model_buffers:
                try:
                    incompatible.missing_keys.remove(k)
                except ValueError:
                    pass
        for k in incompatible.unexpected_keys[:]:
            # Ignore unexpected keys about cell anchors. They exist in old checkpoints
            # but now they are non-persistent buffers and will not be in new checkpoints.
            if "anchor_generator.cell_anchors" in k:
                incompatible.unexpected_keys.remove(k)
        return incompatible

class DefaultPredictor:
    """ predictor """
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        self.checkpointer = DetectionCheckpointer(self.model)

        # --------------------------------------------------------------------------------------------------------------
        # loading model weights
        #self.checkpointer.load(path = cfg.MODEL.WEIGHTS)

        # --------------------------------------------------------------------------------------------------------------
        self.aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image,
                      "height": height,
                      "width": width}
            #predictions = self.model([inputs],self.i)[0]
            predictions = self.model([inputs])[0]
            return predictions

class VisualizationDemo(object):

    def __init__(self,
                 cfg,
                 instance_mode=ColorMode.IMAGE,
                 parallel=False):

        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)
            # model = demo.predictor.model
        self.threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST  # workaround

    def run_on_image(self, image):
        # ------------------------------------------------------------------------------------------------------
        # 1)
        vis_output = None
        predictions = self.predictor(image)
        instances = predictions['instances']

        # ------------------------------------------------------------------------------------------------------
        # 2) filtering
        predictions = {'instances': instances[instances.scores > self.threshold]}

        # ------------------------------------------------------------------------------------------------------
        # 3) convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info)
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(predictions["sem_seg"].argmax(dim=0).to(self.cpu_device))
            if "instances" in predictions:
                instances = instances.to(self.cpu_device)
                #vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return predictions #, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # Converts Matplotlib RGB format to OpenCV BGR format
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))

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
    print(cfg)

    print(f'\n step 3. set demo pipeline')
    detection_pipeline = VisualizationDemo(cfg)
    # ------------------------------------------------------------------------------------------------------------------
    # DefaultPredictor
    print(f' (3.1) make scratch model')
    predictor = detection_pipeline.predictor
    model = predictor.model
    print(f' (3.2) loading pretrained model weights')
    checkpointer = predictor.checkpointer #load(path=cfg.MODEL.WEIGHTS)
    #checkpointer.load(path=cfg.MODEL.WEIGHTS)
    print(f'model name = DiffusionDet')
    component_models = model.named_children()
    for name, component_model in component_models :
        print(f'  - {name} : {component_model.__class__.__name__}')

    head_model = model.head         # DynamicHead
    criterion = model.criterion     # SetCriterionDynamicK

    print(f'\n step 4. check FPN model')
    backbone_model = model.backbone  # FPN
    FPN_component_models = backbone_model.named_children()

    print(f'\n step 5. image inference')

    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        print(f'(4.1) read image')
        np_img = read_image(path, format="BGR")
        predictions = detection_pipeline.run_on_image(np_img)
        """
        batched_inputs = 

        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            feature = src[f]
            features.append(feature)
        """



    """
    for name, component_model in FPN_component_models :
        print(f'  - [FPN] {name} : {component_model.__class__.__name__}')    
      - [FPN] fpn_lateral2 : Conv2d
      - [FPN] fpn_output2 : Conv2d
      - [FPN] fpn_lateral3 : Conv2d
      - [FPN] fpn_output3 : Conv2d
      - [FPN] fpn_lateral4 : Conv2d
      - [FPN] fpn_output4 : Conv2d
      - [FPN] fpn_lateral5 : Conv2d
      - [FPN] fpn_output5 : Conv2d
      - [FPN] top_block : LastLevelMaxPool
      - [FPN] bottom_up : SwinTransformer
    """

    """
    print(f'\n step 4. inference')
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        print(f'(4.1) read image')
        np_img = read_image(path, format="BGR")
        start_time = time.time()
        print(f'(4.2) detecting on one image')
        predictions = detection_pipeline.run_on_image(np_img)
    """


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file",
                        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
                        metavar="FILE", help="path to config file",)
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+", type=str,
                        help="A list of space separated input images; or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",
                        help="A file or directory to save output visualizations. If not given, will show output in an OpenCV window.",)
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    main(args)