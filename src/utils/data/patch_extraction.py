from typing import Any

import numpy as np
from skimage.transform import resize
import torch
from tqdm import tqdm
from torchvision import tv_tensors as tvt
from torchvision.transforms import v2 as T

__all__ = ["PatchView"]


class PatchView:
    """A class representing a view of an image as a collection of patches.

    Access patches through the [] operator.

    Args:
        image (array-like): The image to be viewed as patches. If the image is 2D, it is assumed to be grayscale.
            If the image is 3D, it is assumed to be RGB, with the last dimension being the color channel.
        positions (array-like): A list of positions of the patches. Each position is a list of 4 integers:
            [x1, y1, x2, y2], where (x1, y1) is the top left corner of the patch and (x2, y2) is the bottom right corner.
    """

    _window_positions_cache = {}

    def __init__(self, image=None, positions=[], image_shape=None, fmt="YXYX"):
        self._image = image
        self.image_shape = image_shape or image.shape
        if isinstance(positions, list):
            if isinstance(positions[0], list):
                positions = np.array(positions)
            elif isinstance(positions[0], np.ndarray):
                positions = np.stack(positions)
        self.fmt = fmt
        self.positions = positions

    def set_fmt(self, fmt):
        assert fmt in ["YXYX", "XYXY"], "Invalid format. Must be one of ['YXYX', 'XYXY']"
        if self.fmt == fmt:
            return
        if fmt == "YXYX":
            self.positions = self.positions[:, [1, 0, 3, 2]]
        else:
            self.positions = self.positions[:, [1, 0, 3, 2]]
        self.fmt = fmt

    def get_position_xyxy(self, idx): 
        if self.fmt == 'YXYX':
            y1, x1, y2, x2 = self.positions[idx]
        else:
            x1, y1, x2, y2 = self.positions[idx]
        return x1, y1, x2, y2

    def __getitem__(self, index):
        x1, y1, x2, y2 = self.get_position_xyxy(index)
        return self._image[y1:y2, x1:x2]

    def __len__(self):
        return len(self.positions)

    def set_image(self, image):
        self._image = image

    def apply_mask(self, mask, threshold=0.5):
        filtered_positions = []
        for x1, y1, x2, y2 in self.positions:
            X, Y = self.image_shape[:2]
            X_mask, Y_mask = mask.shape[:2]

            # if the mask is of a different shape than the image,
            # we need to adjust the coordinates to be relative to the mask
            if X != X_mask:
                x1_mask = int(x1 / X * X_mask)
                x2_mask = int(x2 / X * X_mask)
            else:
                x1_mask, x2_mask = x1, x2
            if Y != Y_mask:
                y1_mask = int(y1 / Y * Y_mask)
                y2_mask = int(y2 / Y * Y_mask)
            else:
                y1_mask, y2_mask = y1, y2

            if np.mean(mask[x1_mask:x2_mask, y1_mask:y2_mask]) >= threshold:
                filtered_positions.append([x1, y1, x2, y2])

        self.positions = np.array(filtered_positions)

    @staticmethod
    def _sliding_window_positions(image_size, window_size, stride, align_to="topleft"):
        """
        Generate a list of positions for a sliding window over an image.

        Args:
            image_size (tuple): The size of the image.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].

        Returns:
            positions (array-like): A list of positions of the patches. Each position is a list of 4 integers:
                [x1, y1, x2, y2], where (x1, y1) is the top left corner of the patch and (x2, y2) is the bottom right corner.
        """

        if (image_size, window_size, stride, align_to) not in PatchView._window_positions_cache:
            if len(image_size) == 2:
                x, y = image_size
            else:
                x, y, _ = image_size

            k1, k2 = window_size
            s1, s2 = stride

            positions = np.mgrid[0 : x - k1 + 1 : s1, 0 : y - k2 + 1 : s2]

            # if the last window is not flush with the image, we may need to offset the image slightly
            lastx, lasty = positions[:, -1, -1]
            lastx += k1
            lasty += k2
            if "bottom" in align_to:
                positions[0, :, :] += x - lastx
            if "right" in align_to:
                positions[1, :, :] += y - lasty

            positions = positions.reshape(2, -1).T
            positions = np.concatenate([positions, positions + window_size], axis=1)

            PatchView._window_positions_cache[(image_size, window_size, stride, align_to)] = positions

        return PatchView._window_positions_cache[(image_size, window_size, stride, align_to)]

    @staticmethod
    def from_sliding_window(
        image=None, window_size=(32, 32), stride=(32, 32), align_to="topleft", masks=[], thresholds=[], image_shape=None
    ):
        """Generate a PatchView from a sliding window over an image.

        This factory method can be used to generate a PatchView from a sliding window over an image.
        The sliding window can be filtered by a list of masks. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.

        Args:
            image (array-like): The image to be viewed as patches.
                If the image is 2D, it is assumed to be grayscale;
                if the image is 3D, it is assumed to be RGB, with the last dimension being the color channel.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].
            masks (array-like): A list of masks to apply to the sliding window. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.
                The masks should be 2-dimensional.
            thresholds (array-like): A list of thresholds for the masks.

        Returns:
            PatchView: A PatchView object.

        """
        image_shape = image_shape or image.shape
        positions = PatchView._sliding_window_positions(
            image_shape, window_size, stride, align_to=align_to
        )

        for mask, threshold in zip(masks, thresholds):
            filtered_positions = []
            for x1, y1, x2, y2 in positions:
                X, Y = image.shape[:2]
                X_mask, Y_mask = mask.shape[:2]

                # if the mask is of a different shape than the image,
                # we need to adjust the coordinates to be relative to the mask
                if X != X_mask:
                    x1_mask = int(x1 / X * X_mask)
                    x2_mask = int(x2 / X * X_mask)
                else:
                    x1_mask, x2_mask = x1, x2
                if Y != Y_mask:
                    y1_mask = int(y1 / Y * Y_mask)
                    y2_mask = int(y2 / Y * Y_mask)
                else:
                    y1_mask, y2_mask = y1, y2

                if np.mean(mask[x1_mask:x2_mask, y1_mask:y2_mask]) >= threshold:
                    filtered_positions.append([x1, y1, x2, y2])

            positions = np.array(filtered_positions)

        pv = PatchView(image, positions, image_shape)
        return pv

    @staticmethod 
    def from_horizontal_mask_center_of_mass(window_size, mask, vertical_stride, image=None, image_size=None):

        image_size = image_size
        if image_size is None and image is not None:
            image_size = image.shape[:2]
        if image_size is None: 
            image_size = mask.shape[:2]

        # compute mask center of mass in the horizontal direction
        H, W = window_size

        com = [] 
        for horizontal_line in mask: 
            indices = np.arange(len(horizontal_line))
            com.append(np.sum(indices[horizontal_line > 0]) / np.sum(horizontal_line))

        positions = []
        top = 0
        while True:
            bottom = top + H
            if bottom > image_size[0]:
                break
            middle_H = int(top + bottom) // 2
            middle_W = com[middle_H]
            if np.isnan(middle_W):
                top += vertical_stride
                continue
            else: 
                middle_W = int(middle_W)

            left = middle_W - W // 2
            right = middle_W + W // 2

            if right > image_size[1]: 
                top += vertical_stride
                continue 
            if left < 0:
                top += vertical_stride
                continue 
        
            positions.append([top, left, bottom, right])
            top += vertical_stride
            

        return PatchView(image, positions, image_size)
    
    @staticmethod
    def from_sliding_window_physical_coordinate(
        image,
        image_physical_size,
        window_physical_size,
        stride_physical_size,
        align_to="topleft",
        masks=[],
        thresholds=[],
    ):
        """Generate a PatchView from a sliding window over an image.

        This factory method can be used to generate a PatchView from a sliding window over an image.
        The sliding window can be filtered by a list of masks. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.

        Args:
            image (array-like): The image to be viewed as patches.
                If the image is 2D, it is assumed to be grayscale;
                if the image is 3D, it is assumed to be RGB, with the last dimension being the color channel.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].
            masks (array-like): A list of masks to apply to the sliding window. If the mean of the mask in a window is greater than the corresponding threshold, the window is kept.
                The masks should be 2-dimensional.
            thresholds (array-like): A list of thresholds for the masks.
            physical_coordinate (array-like): The physical coordinate of the image. If None, the physical coordinate is assumed to be the same as the image.

        Returns:
            PatchView: A PatchView object.

        """

        H_px, Y_px = image.shape[:2]
        H_cm, Y_cm = image_physical_size

        window_size = (
            int(window_physical_size[0] / H_cm * H_px),
            int(window_physical_size[1] / Y_cm * Y_px),
        )
        stride = (
            int(stride_physical_size[0] / H_cm * H_px),
            int(stride_physical_size[1] / Y_cm * Y_px),
        )

        return PatchView.from_sliding_window(
            image,
            window_size,
            stride,
            align_to=align_to,
            masks=masks,
            thresholds=thresholds,
        )

    @staticmethod
    def build_collection_from_images_and_masks(
        window_size,
        stride,
        image_list=None,
        align_to="topleft",
        mask_lists=[],
        thresholds=[],
        image_shape=None,
    ):
        """Generate a collection of PatchViews from a collection of images and masks.

        Because this will vectorize the mask intersection calculations, it is much faster than calling from_sliding_window multiple times.
        However, this method requires that all images and masks are of the same size.

        Args:
            image_list (array-like): A list of images to be viewed as patches.
                If the images are 2D, they are assumed to be grayscale;
                if the images are 3D, they are assumed to be RGB, with the last dimension being the color channel.
                if you don't want to provide images at time of calling, you can pass None and provide image_shape.
            window_size (tuple): The size of the sliding window.
            stride (tuple): The stride of the sliding window.
            align_to (str): The alignment of the sliding window. Can be one of: ['topleft', 'bottomleft', 'topright', 'bottomright'].
            mask_lists (array-like): A list of lists of masks to apply to the sliding window. If the mean of the mask in a window is greater
                than the corresponding threshold, the window is kept. The masks should be 2-dimensional. If more then one list of masks is provided,
                they will be applied in order to filter the windows.
            thresholds (array-like): A list of thresholds for the masks.
        """
        if image_list is None: 
            image_list = [None] * len(mask_lists[0])

        n_images = len(image_list)
        H, W = image_shape or image_list[0].shape[:2]
        position_candidates = PatchView._sliding_window_positions(
            image_shape, window_size, stride, align_to=align_to
        )

        n_position_candidates = len(position_candidates)
        valid_position_candidates = np.ones(
            (n_images, n_position_candidates), dtype=bool
        )

        for mask_list, threshold in zip(mask_lists, thresholds):
            valid_position_candidates_for_mask = np.zeros(
                (n_images, n_position_candidates), dtype=bool
            )
            mask_arr = np.stack(mask_list, axis=-1)

            for idx in tqdm(range(n_position_candidates), desc="Applying mask"):
                x1, y1, x2, y2 = position_candidates[idx]
                x1 = int(x1 / H * mask_arr.shape[0])
                x2 = int(x2 / H * mask_arr.shape[0])
                y1 = int(y1 / W * mask_arr.shape[1])
                y2 = int(y2 / W * mask_arr.shape[1])

                valid_position_candidates_for_mask[:, idx] = (
                    mask_arr[x1:x2, y1:y2].mean(axis=(0, 1)) > threshold
                )

            valid_position_candidates *= valid_position_candidates_for_mask

        patch_views = []
        for idx in tqdm(range(n_images), desc="Filtering positions"):
            positions_for_image = []
            for j in range(n_position_candidates):
                if valid_position_candidates[idx, j]:
                    position = position_candidates[j]
                    positions_for_image.append(position)

            patch_views.append(PatchView(image_list[idx], positions_for_image, image_shape=image_shape))

        return patch_views

    def show(self, highlight_idx=None, ax=None, **kwargs):
        """Show the patch view by plotting the patches on top of the image.

        Args:
            ax (matplotlib.axes.Axes): The axes to plot on. If None, a new figure and axes is created.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        image = self._image
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None, **kwargs)

        for idx in range(len(self)):
            y1, x1, y2, x2 = self.get_position_xyxy(idx)
            ax.plot([y1, y2, y2, y1, y1], [x1, x1, x2, x2, x1], "r")

        if highlight_idx is not None:
            y1, x1, y2, x2 = self.get_position_xyxy(highlight_idx)
            ax.plot([y1, y2, y2, y1, y1], [x1, x1, x2, x2, x1], "g", linewidth=2)

        ax.axis("off")
        return ax

    def crop(self, top, left, height, width): 
        """Crop the PatchView to a new region of interest."""
        prev_fmt = self.fmt

        self.set_fmt('XYXY')
        bboxes = torch.tensor(self.positions)

        adjusted_bboxes, canvas_size = T.functional.crop_bounding_boxes(
            bboxes, tvt.BoundingBoxFormat.XYXY, top, left, height, width
        )
        self.positions = adjusted_bboxes.numpy()

        if self.image_shape is not None:
            self.image_shape = (height, width)
        if self._image is not None:
            self._image = self._image[top:top+height, left:left+width]

        self.set_fmt(prev_fmt)
        return self

    def resize(self, height, width): 
        prev_fmt = self.fmt
        self.set_fmt('XYXY')
        bboxes = torch.tensor(self.positions)

        canvas_size = self.image_shape or self._image.shape[:2]

        adjusted_bboxes, canvas_size = T.functional.resize_bounding_boxes(
            bboxes, canvas_size, (height, width)
        )

        self.positions = adjusted_bboxes.numpy()

        if self.image_shape is not None:
            self.image_shape = (height, width)

        if self._image is not None:
            self._image = resize(self._image, (height, width))

        self.set_fmt(prev_fmt)
        return self


