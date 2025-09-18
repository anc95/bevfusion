"""
torchpack dist-run -np 1 python tools/visualize-at128-with-image.py configs/nuscenes/seg/fusion-bev256d2-lss.yaml --mode pred --checkpoint pretrained/bevfusion-seg.pth
"""

import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel, DataContainer
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
# from torchpack.utils.tqdm import tqdm
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()

def isolate_single_image(batch_tensor, keep_index=0):
    assert batch_tensor.shape[1] == 6
    assert 0 <= keep_index < 6
    output = torch.zeros_like(batch_tensor)
    output[0, keep_index] = batch_tensor[0, keep_index]
    return output

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def prepare_data(dataset, lidar_path, image_path):
    lidar2camera = np.array([
        [-0.0129958,  0.031462,  -0.99942,    0.319262],
        [-0.0215808,  0.999263,   0.0317376, -0.00363188],
        [ 0.999683,   0.0219808, -0.0123072,  0.141231],
        [ 0,          0,          0,          1]
    ])

    camera_intrinsics = np.array([
        [1.23569864e+03, 0.00000000e+00, 2.01395301e+03, 0.00000000e+00],
        [0.00000000e+00, 1.23578552e+03, 1.51927919e+03, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
        [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

    lidar2image = camera_intrinsics @ lidar2camera

    input_dict = {
        "token": "000000",
        "sample_idx": "000000",
        "lidar_path": lidar_path,
        "image_paths": [image_path],
        "timestamp": 1530000000000000,
        "ego2global": np.eye(4),
        "lidar2ego": np.eye(4),
        "lidar2camera": [lidar2camera],
        "lidar2image": [lidar2image],
        "camera2ego": [np.linalg.inv(lidar2camera)],
        "camera2lidar": [np.linalg.inv(lidar2image)],
        "camera_intrinsics": [camera_intrinsics],
        "sweeps": [],
        "location": "beijing",
        "metas": {
            "filename": [image_path],
            "timestamp": 1530000000000000,
            "token": "000000",
            "sample_idx": "000000",
            "lidar2image": [lidar2image],
            "lidar2camera": [lidar2camera],
            "camera2ego": [np.linalg.inv(lidar2camera)],
        }
    }

    dataset.pre_pipeline(input_dict)
    example = dataset.pipeline(input_dict)
    return example


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--lidar-path", type=str, default=None)
    parser.add_argument("--image-path", type=str, default=None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    data = prepare_data(dataset, args.lidar_path, args.image_path)
    metas = data["metas"]
    name = "{}-{}".format(metas["timestamp"], metas["token"])

    if args.mode == "pred":
        with torch.inference_mode():
            outputs = model(**data, depths=None)

    if args.mode == "gt" and "gt_bboxes_3d" in data:
        bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        labels = data["gt_labels_3d"].data[0][0].numpy()

        if args.bbox_classes is not None:
            indices = np.isin(labels, args.bbox_classes)
            bboxes = bboxes[indices]
            labels = labels[indices]

        bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    elif args.mode == "pred" and "boxes_3d" in outputs[0]:
        bboxes = outputs[0]["boxes_3d"].tensor.numpy()
        scores = outputs[0]["scores_3d"].numpy()
        labels = outputs[0]["labels_3d"].numpy()

        if args.bbox_classes is not None:
            indices = np.isin(labels, args.bbox_classes)
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        if args.bbox_score is not None:
            indices = scores >= args.bbox_score
            bboxes = bboxes[indices]
            scores = scores[indices]
            labels = labels[indices]

        bboxes[..., 2] -= bboxes[..., 5] / 2
        bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
    else:
        bboxes = None
        labels = None

    if args.mode == "gt" and "gt_masks_bev" in data:
        masks = data["gt_masks_bev"].data[0].numpy()
        masks = masks.astype(np.bool)
    elif args.mode == "pred" and "masks_bev" in outputs[0]:
        masks = outputs[0]["masks_bev"].numpy()
        masks = masks >= args.map_score
    else:
        masks = None

    if "img" in data:
        for k, image_path in enumerate(metas["filename"]):
            image = mmcv.imread(image_path)
            visualize_camera(
                os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
                image,
                bboxes=bboxes,
                labels=labels,
                transform=metas["lidar2image"][k],
                classes=cfg.object_classes,
            )

    if "points" in data:
        lidar = data["points"].data[0][0].numpy()
        visualize_lidar(
            os.path.join(args.out_dir, "lidar", f"{name}.png"),
            lidar,
            bboxes=bboxes,
            labels=labels,
            xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
            ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
            classes=cfg.object_classes,
        )

    if masks is not None:
        visualize_map(
            os.path.join(args.out_dir, "map", f"{name}.png"),
            masks,
            classes=cfg.map_classes,
        )


if __name__ == "__main__":
    main()
