# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import hydra
import torch
from omegaconf import OmegaConf, DictConfig

from vggsfm.utils.utils import seed_all_random_engines
from vggsfm.runners.runner import VGGSfMRunner
from vggsfm.datasets.demo_loader import DemoLoader


@torch.no_grad()
@hydra.main(config_path="cfgs/", config_name="demo", version_base="1.2")
def demo_fn(cfg: DictConfig):
    """
    Main function to run the VGGSfM demo. VGGSfMRunner is the main controller.
    """

    # Print configuration
    OmegaConf.set_struct(cfg, False)
    print("Model Config:", OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    seed_all_random_engines(cfg.seed)

    # Initialize VGGSfM Runner
    vggsfm_runner = VGGSfMRunner(cfg)

    # Load Data
    test_dataset = DemoLoader(
        SCENE_DIR=cfg.SCENE_DIR,
        img_size=cfg.img_size,
        normalize_cameras=False,
        load_gt=cfg.load_gt,
    )

    sequence_list = test_dataset.sequence_list

    seq_name = sequence_list[0]  # Run on one Scene

    # Load the data for the selected sequence
    batch, image_paths = test_dataset.get_data(
        sequence_name=seq_name,
        return_path=True,
    )

    output_dir = batch["scene_dir"]  # which is also cfg.SCENE_DIR for DemoLoader
    images = batch["image"]
    masks = batch["masks"] if batch["masks"] is not None else None
    crop_params = batch["crop_params"] if batch["crop_params"] is not None else None

    # Cache the original images for visualization, so that we don't need to re-load many times
    # original_images = batch["original_images"]

    # Run VGGSfM
    # Both visualization and output writing are performed inside VGGSfMRunner
    vggsfm_runner.run(
        images,
        masks=masks,
        # original_images=original_images,
        image_paths=image_paths,
        crop_params=crop_params,
        seq_name=seq_name,
        output_dir=output_dir,
    )

    print("Demo Finished Successfully")


if __name__ == "__main__":
    demo_fn()
