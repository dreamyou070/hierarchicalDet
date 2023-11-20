import argparse
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from diffusiondet.predictor import VisualizationDemo
from diffusiondet import DiffusionDetDatasetMapper, add_diffusiondet_config, DiffusionDetWithTTA
from diffusiondet.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, apply_model_ema_and_restore, EMADetectionCheckpointer
from PIL import Image
import cv2


def main(args) :

    print(f'step1. setup config')
    print(f' (1) base config')
    cfg = get_cfg()
    print(f' (2) diffusiondet config (RCNN Head., Dynamic Conv., Loss., Focal Loss., Diffusion,  Swin Backbones)')
    add_diffusiondet_config(cfg)
    print(f' (3) model ema config (ema means copy the model, little decay)')
    add_model_ema_configs(cfg)
    print(f' (4) read config file')
    cfg.merge_from_file(args.config_file)
    print(f' (5) add opt config')
    cfg.merge_from_list(args.opts)
    print(f' (6) set score_threshold for builtin models')
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()

    print(f'step2. make demo pipeline')
    demo = VisualizationDemo(cfg)

    """    
    predictor = demo.predictor
    main(predictor)
    img_dir = r'../../563.JPG'
    cv2_img = cv2.imread(img_dir)
    output = predictor(cv2_img)
    instsance_info = output['instances']
    """

if __name__ == "__main__":
    #mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument("--config-file",default="configs/diffdet.coco.swinbase.yaml",)
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument("--input", nargs="+",
                        help="A list of space separated input images; or a single glob pattern such as 'directory/*.jpg'",)
    parser.add_argument("--output",help="A file or directory to save output visualizations.")
    parser.add_argument("--confidence-threshold",type=float,default=0.5,
                        help="Minimum score for instance predictions to be shown",)
    parser.add_argument("--opts",help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[],nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    main(args)


    """
    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            
            predictions, visualized_output = demo.run_on_image(img)
            logger.info("{}: {} in {:.2f}s".format(path,
                                                   "detected {} instances".format(len(predictions["instances"]))
                                                   if "instances" in predictions
                                                   else "finished",
                                                   time.time() - start_time,))

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
    """
