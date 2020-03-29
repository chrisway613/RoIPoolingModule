import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIAlign(nn.Module):
    """RoIAlign Implementation"""

    def __init__(self, spacial_scale=1. / 32, pooled_height=7, pooled_width=7, num_sample=4):
        """
        :param spacial_scale: scale ratio -- size_feature / size_roi
        :param pooled_height: target height after pooling
        :param pooled_width: target width after pooling
        """
        super(RoIAlign, self).__init__()

        self.spacial_scale = spacial_scale
        self.pooled_height = pooled_height
        self.pooled_width = pooled_width
        self.num_sample = num_sample

    def forward(self, features, rois):
        _, C, H, W = features.shape
        num_roi = rois.shape[0]
        pooled = torch.zeros(num_roi, C, self.pooled_height, self.pooled_width)

        for idx, roi in enumerate(rois):
            # no quantization
            bbox = roi[1:].numpy() * self.spacial_scale
            # clip to the range of the size of feature map
            np.clip(bbox[0::2], 0, W - 1, out=bbox[0::2])
            np.clip(bbox[1::2], 0, H - 1, out=bbox[1::2])
            # print("bbox:", bbox)

            # size of each bin
            bin_w = (bbox[2] - bbox[0]) / self.pooled_width
            bin_h = (bbox[3] - bbox[1]) / self.pooled_height
            # print("bin width:", bin_w)
            # print("bin_height:", bin_h)

            for i in range(self.pooled_height):
                # no quantization
                bin_start_y = bbox[1] + i * bin_h
                bin_end_y = bbox[1] + (i + 1) * bin_h

                # clip to [0, H - 1]
                bin_start_y = np.clip(bin_start_y, 0, H - 1)
                bin_end_y = np.clip(bin_end_y, 0, H - 1)

                for j in range(self.pooled_width):
                    # No quantization
                    bin_start_x = bbox[0] + j * bin_w
                    bin_end_x = bbox[0] + (j + 1) * bin_w

                    # clip to [0, W - 1]
                    bin_start_x = np.clip(bin_start_x, 0, W - 1)
                    bin_end_x = np.clip(bin_end_x, 0, W - 1)

                    if bin_start_y < bin_end_y and bin_start_x < bin_end_x:
                        k = int(np.sqrt(self.num_sample))
                        sub_bin_w, sub_bin_h = bin_w / k, bin_h / k
                        # center point position of top left sub bin
                        sub_bin_tl_xc = bin_start_x + sub_bin_w / 2
                        sub_bin_tl_yc = bin_start_y + sub_bin_h / 2
                        # center point position of each sub bin
                        sub_bin_c = np.zeros((k, k, 2))

                        for m in range(k):
                            yc = sub_bin_tl_yc + m * sub_bin_h

                            for n in range(k):
                                xc = sub_bin_tl_xc + n * sub_bin_w
                                sub_bin_c[m, n] = [xc, yc]

                        batch_index = int(roi[0].item())
                        f = features[batch_index]
                        # interpolated value on target position
                        # (C, k, k)
                        # print("feature:", f)
                        # print("sub bin center:", sub_bin_c)
                        interpolated_f = self.interpolation(f, sub_bin_c)
                        # print("interpolated:", interpolated_f)
                        p = pooled[idx]
                        # max pooling
                        p[:, i, j] = F.adaptive_max_pool2d(interpolated_f, 1).squeeze()

        return pooled

    @staticmethod
    def interpolation(feature, pos):
        out_c, H, W = feature.shape
        out_h, out_w = pos.shape[:2]
        out = torch.zeros((out_c, out_h, out_w))

        for i in range(out_h):
            for j in range(out_w):
                x, y = pos[i][j][0], pos[i][j][1]
                x0, y0 = int(np.floor(x)), int(np.floor(y))
                x0, y0 = np.clip(x0, 0, W - 2), np.clip(y0, 0, H - 2)
                x1, y1 = x0 + 1, y0 + 1

                # pixel value of 4 adjacent point
                # left top, right top, left bottom, right bottom
                lt, rt, lb, rb = feature[:, y0, x0], feature[:, y0, x1], feature[:, y1, x0], feature[:, y1, x1]
                # interpolation in x direction
                # middle top, middle bottom
                mt, mb = (x - x0) * rt + (x1 - x) * lt, (x - x0) * rb + (x1 - x) * lb
                # interpolation in y direction
                pixel_xy = (y - y0) * mb + (y1 - y) * mt
                out[:, i, j] = pixel_xy

        return out
