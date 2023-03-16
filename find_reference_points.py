import cv2
import numpy as np


def unwarp(img, src, dst):
    h, w = img.shape[:2]
    # use cv2.getPerspectiveTransform() to get M, the transform matrix, and Minv, the inverse
    M = cv2.getPerspectiveTransform(src, dst)
    # use cv2.warpPerspective() to warp your image to a top-down view
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
    return warped


template = cv2.imread('./images/reference.JPG')
large_image = cv2.imread('./images/image.JPG')

lighter_red_up = np.array([40, 40, 100])
lighter_red_low = np.array([90, 90, 220])
lighter_red_mask = cv2.inRange(large_image, lighter_red_up, lighter_red_low)

darker_red_up = np.array([90, 90, 150])
darker_red_low = np.array([130, 130, 230])
darker_red_mask = cv2.inRange(large_image, darker_red_up, darker_red_low)

red_mask = lighter_red_mask + darker_red_mask
# image_width, image_height = large_image.shape[:2]
# image_offsets = [
#     [0, image_width/2, 0, image_height/2],
#     [0, image_width/2, image_height/2, image_height],
#     [image_width/2, image_width, 0, image_height/2],
#     [image_width/2, image_width, image_height/2, image_height]
# ]
# for region_boundaries in image_offsets:
#     [y_start, y_end, x_start, x_end] = list(map(int, region_boundaries))
#     region = large_image[y_start:y_end, x_start:x_end]
#     # cv2.rectangle(large_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)
#     mask = cv2.inRange(region, lower_red, upper_red)
#     j = 0
#     leftmost_points = []
#     for i in range(len(mask)):
#         nonempty = [(x, i) for (i, x) in enumerate(mask[i]) if x > 0]
#         if len(nonempty) > 0:
#             (_, x) = nonempty[0]
#             leftmost_points.append([x, i])
#     sorted_list = sorted(leftmost_points, key=lambda x: x[0])
#     x, y = sorted_list[0]
#     cv2.rectangle(large_image, (x_start + x, y_start + y), (x_start + x + 20, y_start + y + 20), (0, 0, 255), 1)
#
# src = np.float32([(20,     1),
#                   (540,  130),
#                   (20,    520),
#                   (570,  450)])
#
# dst = np.float32([(0, 0),
#                   (600, 0),
#                   (0, 600),
#                   (600, 600)])
#
# final_image = unwarp(large_image, src, dst)

contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
im = np.copy(large_image)
cv2.drawContours(im, contours, -1, (0, 255, 0), 1)
boxes = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 5000:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)
        # cv2.drawContours(large_image, [box], 0, (0, 0, 255), 2)
# cv2.imwrite("contours_blue.png", im)
# cv2.imshow("output", red_mask)
# cv2.imshow("output", im)
# Display the original image with the rectangle around the match.
boxes = sorted(boxes,  key=lambda el: (el[0][1], el[0][0]))
corners = []
for i in range(4):
    corners.append(boxes[i][i])
corners = np.array(corners)
print(type(corners), type(corners[0]))
cv2.drawContours(im, [corners], 0, (0, 0, 255), 2)


src = np.float32([corners])

dst = np.float32([(0, 0),
                  (0, 1000),
                  (1000, 1000),
                  (1000, 0),
                  ])

final_image = unwarp(large_image, src, dst)
final_image = final_image[100:900, 100:900]

# cv2.imshow('original', large_image)
cv2.imshow('box', im)
# cv2.imshow('output', final_image)

# The image is only displayed if we call this
cv2.waitKey(0)
