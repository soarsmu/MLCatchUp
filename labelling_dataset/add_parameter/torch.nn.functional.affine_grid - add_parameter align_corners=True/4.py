from torch import nn
import cv2
import numpy as np
import math
import torch
import skimage.transform as trans
from ...utils.util import show_box

class ROIRotate(nn.Module):

    def __init__(self, height=8):
        '''

        :param height: heigth of feature map after affine transformation
        '''

        super().__init__()
        self.height = height

    def forward(self, feature_map, boxes, mapping):
        '''

        :param feature_map:  N * 128 * 128 * 32
        :param boxes: M * 8
        :param mapping: mapping for image
        :return: N * H * W * C
        '''

        max_width = 0
        boxes_width = []
        matrixes = []
        images = []

        for img_index, box in zip(mapping, boxes):
            feature = feature_map[img_index]  # B * H * W * C
            images.append(feature)

            x1, y1, x2, y2, x3, y3, x4, y4 = box / 4  # 521 -> 128


            # show_box(feature, box / 4, 'ffffff', isFeaturemap=True)

            rotated_rect = cv2.minAreaRect(np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]))
            box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

            width = feature.shape[2]
            height = feature.shape[1]

            if box_w <= box_h:
                box_w, box_h = box_h, box_w

            mapped_x1, mapped_y1 = (0, 0)
            mapped_x4, mapped_y4 = (0, self.height)

            width_box = math.ceil(self.height * box_w / box_h)
            width_box = min(width_box, width) # not to exceed feature map's width
            max_width = width_box if width_box > max_width else max_width

            mapped_x2, mapped_y2 = (width_box, 0)

            src_pts = np.float32([(x1, y1), (x2, y2),(x4, y4)])
            dst_pts = np.float32([
                (mapped_x1, mapped_y1), (mapped_x2, mapped_y2), (mapped_x4, mapped_y4)
            ])

            affine_matrix = cv2.getAffineTransform(src_pts.astype(np.float32), dst_pts.astype(np.float32))

            # iii = cv2.warpAffine((feature*255).permute(1,2,0).detach().cpu().numpy().astype(np.uint8),
            #                      affine_matrix, (width, height))
            #
            # cv2.imshow('img', iii)
            # cv2.waitKey()

            affine_matrix = ROIRotate.param2theta(affine_matrix, width, height)

            affine_matrix *= 1e20 # cancel the error when type conversion
            affine_matrix = torch.tensor(affine_matrix, device=feature.device, dtype=torch.float)
            affine_matrix /= 1e20

            matrixes.append(affine_matrix)
            boxes_width.append(width_box)

        matrixes = torch.stack(matrixes)
        images = torch.stack(images)
        grid = nn.functional.affine_grid(matrixes, images.size())
        feature_rotated = nn.functional.grid_sample(images, grid)

        channels = feature_rotated.shape[1]
        cropped_images_padded = torch.zeros((len(feature_rotated), channels, self.height, max_width),
                                            dtype=feature_rotated.dtype,
                     