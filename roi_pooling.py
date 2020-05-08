import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPooling2D(nn.Module):
    """RoIPooling2D Implementation"""

    def __init__(self, spacial_scale=1. / 32, pooled_height=7, pooled_width=7):
        """
        :param spacial_scale: scale ratio -- size_feature / size_roi
        :param pooled_height: target height after pooling
        :param pooled_width: target width after pooling
        """
        super(RoIPooling2D, self).__init__()

        self.spacial_scale = spacial_scale
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def forward(self, features, rois):
        """
        RoIModule 2D.
        :param features: (N, C, H, W) -> a batch of feature maps
        :param rois: (num_roi, 5) -> each roi is (feature_index_of_batch, x_min, y_min, x_max, y_max)
        :return: pooled target with shape (num_roi, C, pooled_height, pooled_width)
        """

        _, C, H, W = features.shape
        num_roi = rois.shape[0]
        # pooled target, project each roi to the corresponding feature map
        pooled = torch.zeros(num_roi, C, self.pooled_height, self.pooled_width)

        for idx, roi in enumerate(rois):
            # 1st quantization
            bbox = np.round(roi[1:].numpy() * self.spacial_scale).astype(int)
            # print("bounding boxes:", bbox)

            bbox_w, bbox_h = max(bbox[2] - bbox[0] + 1, 1), max(bbox[3] - bbox[1] + 1, 1)
            bin_w, bin_h = float(bbox_w) / self.pooled_width, float(bbox_h) / self.pooled_height

            for i in range(self.pooled_height):
                # 2nd quantization
                y_start = bbox[1] + int(np.floor(i * bin_h))
                y_end = bbox[1] + int(np.ceil((i + 1) * bin_h))
                # print("y_start:", y_start)
                # print("y_end:", y_end)

                # clip to range [0, H - 1]
                y_start = np.clip(y_start, 0, H - 1)
                y_end = np.clip(y_end, 0, H - 1)

                for j in range(self.pooled_width):
                    x_start = bbox[0] + int(np.floor(j * bin_w))
                    x_end = bbox[0] + int(np.ceil((j + 1) * bin_w))
                    #                     print("x_start:", x_start)
                    #                     print("x_end:", x_end)

                    # clip to range [0, W - 1]
                    x_start = np.clip(x_start, 0, W - 1)
                    x_end = np.clip(x_end, 0, W - 1)

                    if y_end > y_start and x_end > x_start:
                        feature_idx = int(roi[0].item())
                        f = features[feature_idx]
                        p = pooled[idx]
                        p[:, i, j] = torch.max(torch.max(f[:, y_start:y_end, x_start:x_end], dim=1)[0], dim=1)[0]

        return pooled


class RoIPooling2DTorch(nn.Module):
    """RoIPooling2D, implemented with torch.nn,functional.adaptive_max_pool2d"""

    def __init__(self, spacial_scale=1. / 32, pooled_height=7, pooled_width=7):
        """
        :param spacial_scale: scale ratio -- size_feature / size_roi
        :param pooled_height: target height after pooling
        :param pooled_width: target width after pooling
        """
        super(RoIPooling2DTorch, self).__init__()

        self.spacial_scale = spacial_scale
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width

    def forward(self, features, rois):
        """
        RoIModule 2D.
        :param features: (N, C, H, W) -> a batch of feature maps
        :param rois: (num_roi, 5) -> each roi is (feature_index_of_batch, x_min, y_min, x_max, y_max)
        :return: pooled target with shape (num_roi, C, pooled_height, pooled_width)
        """
        _, C, H, W = features.shape
        num_roi = rois.shape[0]
        # pooled target, project each roi to the corresponding feature map
        pooled = torch.zeros(num_roi, C, self.pooled_height, self.pooled_width)

        for idx, roi in enumerate(rois):
            # quantization
            bbox = np.round(roi[1:].numpy() * self.spacial_scale).astype(np.int)
            # clip to the range of feature size
            np.clip(bbox[0::2], 0, W - 1, out=bbox[0::2])
            np.clip(bbox[1::2], 0, H - 1, out=bbox[1::2])

            # print("bounding box:", bbox)

            if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                feature_idx = int(roi[0].item())
                f = features[feature_idx]
                projected_roi = f[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # Max Pooling
                pooled_projected_roi = F.adaptive_max_pool2d(
                    projected_roi,
                    (self.pooled_height, self.pooled_height)
                )
                pooled[idx] = pooled_projected_roi

        return pooled
