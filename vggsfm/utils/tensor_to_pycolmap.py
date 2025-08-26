# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
import pycolmap
from pycolmap import CameraModelId

GLOBAL_RIG_ID: int = 10001


def batch_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    masks,
    image_size,
    max_points3D_val: int = 3000,
    shared_camera: bool = False,
    camera_type=CameraModelId.SIMPLE_PINHOLE,
    extra_params: torch.Tensor | None = None,
) -> pycolmap.Reconstruction:
    """
    Convert Batched Pytorch Tensors to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format
    """

    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    extrinsics = extrinsics.detach().cpu().numpy()
    intrinsics = intrinsics.detach().cpu().numpy()

    if extra_params is not None:
        assert isinstance(extra_params, torch.Tensor)
        extra_params = extra_params.detach().cpu().numpy()

    tracks = tracks.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    points3d = points3d.detach().cpu().numpy()
    image_size = image_size.detach().cpu().numpy()

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    # Only add 3D points that have sufficient 2D points
    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]
    for vidx in valid_idx:
        reconstruction.add_point3D(
            points3d[vidx],
            pycolmap.Track(),
            np.zeros(3, dtype=np.uint8),
        )

    num_points3D = len(valid_idx)

    # set rig
    rig = pycolmap.Rig(rig_id=GLOBAL_RIG_ID)
    reconstruction.add_rig(rig)

    # frame idx
    camera = None
    for fidx in range(N):
        # set camera
        if (camera is None) or (rig is None) or (not shared_camera):
            if camera_type == CameraModelId.SIMPLE_RADIAL:
                assert extra_params is not None
                params = np.array(
                    [
                        intrinsics[fidx][0, 0].item(),
                        intrinsics[fidx][0, 2].item(),
                        intrinsics[fidx][1, 2].item(),
                        extra_params[fidx][0].item(),
                    ],
                    dtype=np.float64,
                )
            elif camera_type == CameraModelId.PINHOLE:
                params = np.array(
                    [
                        intrinsics[fidx][0, 0].item(),
                        intrinsics[fidx][0, 1].item(),
                        intrinsics[fidx][0, 2].item(),
                        intrinsics[fidx][1, 2].item(),
                    ],
                    dtype=np.float64,
                )
            elif camera_type == CameraModelId.SIMPLE_PINHOLE:
                params = np.array(
                    [
                        intrinsics[fidx][0, 0].item(),
                        intrinsics[fidx][0, 2].item(),
                        intrinsics[fidx][1, 2].item(),
                    ],
                    dtype=np.float64,
                )
            else:
                raise ValueError(f'Camera type {camera_type} is not supported yet')

            camera = pycolmap.Camera(
                camera_id=fidx,
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=params,
            )

            # add camera
            reconstruction.add_camera(camera)

            # set sensor
            sensor_t = pycolmap.sensor_t()
            sensor_t.id = camera.camera_id
            sensor_t.type = pycolmap.SensorType.CAMERA

            if reconstruction.rig(GLOBAL_RIG_ID).num_sensors() == 0:
                reconstruction.rig(GLOBAL_RIG_ID).add_ref_sensor(sensor_t)
            else:
                reconstruction.rig(GLOBAL_RIG_ID).add_sensor(sensor_t, pycolmap.Rigid3d())

        # set frame
        pose = extrinsics[fidx].astype(np.float64)
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(pose[:3, :3]), pose[:3, 3])

        # set frame
        frame = pycolmap.Frame(
            frame_id=fidx,
            rig_id=rig.rig_id,
        )

        # set image
        image = pycolmap.Image(
            image_id=fidx,
            name=f'image_{fidx:04d}',
            frame_id=frame.frame_id,
            camera_id=camera.camera_id,
        )
        frame.add_data_id(image.data_id)
        reconstruction.add_frame(frame)
        reconstruction.frame(frame.frame_id).set_cam_from_world(camera.camera_id, cam_from_world)

        # NOTE: point3D_id start by 1
        points2D_list = []
        point2D_idx = 0
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]
            if (reconstruction.point3D(point3D_id).xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx, point2D_idx)
                    point2D_idx += 1

        try:
            image.points2D = pycolmap.Point2DList(points2D_list)
            reconstruction.register_image(image.frame_id)
        except Exception as e:
            logging.warning(f'frame {fidx} is out of BA: {e}')
            reconstruction.deregister_image(image.frame_id)

        reconstruction.add_image(image)

    return reconstruction


def pycolmap_to_batch_matrix(
    reconstruction: pycolmap.Reconstruction,
    device: str = 'cuda',
    camera_type=CameraModelId.SIMPLE_PINHOLE,
):
    """
    Convert a PyCOLMAP Reconstruction Object to batched PyTorch tensors.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): The device to place the tensors on (default: "cuda").
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    num_points3D = len(reconstruction.points3D)
    if num_points3D == 0:
        raise Exception('There are no points remained...')

    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3), dtype=np.float64)
    for i in range(num_points3D):
        points3D[i] = reconstruction.point3D(i + 1).xyz

    points3D = torch.Tensor(points3D).to(device=device)

    extrinsics = []
    intrinsics = []
    extra_params = [] if camera_type == CameraModelId.SIMPLE_RADIAL else None

    for fidx in range(num_images):
        # Extract and append extrinsics
        img = reconstruction.image(fidx)
        cam = reconstruction.camera(img.camera_id)
        matrix = img.cam_from_world().matrix()

        assert isinstance(matrix, np.ndarray)
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = cam.calibration_matrix()
        assert isinstance(calibration_matrix, np.ndarray)
        intrinsics.append(calibration_matrix)

        if extra_params is not None:
            extra_params.append(cam.params[-1])

    # Convert lists to torch tensors
    extrinsics = torch.Tensor(np.stack(extrinsics, axis=0)).to(device)
    intrinsics = torch.Tensor(np.stack(intrinsics, axis=0)).to(device)
    if extra_params is not None:
        extra_params = torch.Tensor(np.stack(extra_params, axis=0)).to(device)
        extra_params = extra_params.unsqueeze(1)

    return points3D, extrinsics, intrinsics, extra_params
