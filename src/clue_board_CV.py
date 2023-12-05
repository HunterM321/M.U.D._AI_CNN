import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import tensorflow as tf

def find_corners(corners):
        # Initialize corners with the first point in the list
        top_left = top_right = bottom_right = bottom_left = corners[0]

        for x, y in corners:
            if x + y < top_left[0] + top_left[1]:
                top_left = (x, y)
            if x - y > top_right[0] - top_right[1]:
                top_right = (x, y)
            if x + y > bottom_right[0] + bottom_right[1]:
                bottom_right = (x, y)
            if x - y < bottom_left[0] - bottom_left[1]:
                bottom_left = (x, y)

        return top_left, top_right, bottom_right, bottom_left

def distance(p1, p2):
    """Calculate the distance between two points."""
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5

def angle(p1, p2, p3):
    """Calculate the angle at p2 formed by the line segments p1-p2 and p2-p3."""
    a = distance(p2, p1)
    b = distance(p2, p3)
    c = distance(p1, p3)
    return math.acos((a**2 + b**2 - c**2) / (2 * a * b)) * (180 / math.pi)

def is_rectangle(tl, tr, br, bl):
    """Check if the points form a rectangle."""
    # Check the distances of opposite sides
    top_length = distance(tl, tr)
    bottom_length = distance(bl, br)
    left_length = distance(tl, bl)
    right_length = distance(tr, br)

    # if not (math.isclose(top_length, bottom_length) and math.isclose(left_length, right_length)):
    #     return False

    # Check the angles
    angles = [
        angle(tl, tr, br),
        angle(tr, br, bl),
        angle(br, bl, tl),
        angle(bl, tl, tr)
    ]
    return all(60 <= a <= 120 for a in angles)  # Allowing some tolerance

def split_texts(img):
    letter_width = 45
    letter_height = 80

    # Define the regions
    pts_0 = np.array([[250, 40], [250, 40 + letter_height], [250 + 6 * letter_width, 40 + letter_height], [250 + 6 * letter_width, 40]], np.int32)
    pts_1 = np.array([[30, 260], [30, 260 + letter_height], [30 + 12 * letter_width, 260 + letter_height], [30 + 12 * letter_width, 260]], np.int32)

    # Split pts_0 into 6 images
    sub_images_pts_0 = []
    for i in range(6):
        x_start = pts_0[0][0] + i * letter_width
        y_start = pts_0[0][1]
        sub_img = img[y_start:y_start + letter_height, x_start:x_start + letter_width]
        sub_images_pts_0.append(sub_img)

    # Split pts_1 into 12 images
    sub_images_pts_1 = []
    for i in range(12):
        x_start = pts_1[0][0] + i * letter_width
        y_start = pts_1[0][1]
        sub_img = img[y_start:y_start + letter_height, x_start:x_start + letter_width]
        sub_images_pts_1.append(sub_img)

    return sub_images_pts_0, sub_images_pts_1

def int2char(n):
    if 0 <= n <= 25:
        return chr(ord('A') + n)
    elif 26 <= n <= 35:
        return chr(ord('0') + n - 26)
    elif n == 36:
        return ' '
    else:
        raise ValueError('Input should be between 0 and 36 inclusive')

def predict_each(images, model):
    predicted_string = ''
    for img in images:
        # Prepare the image for model
        img = np.expand_dims(img, axis=0)

        # Predict the letter
        prediction = model.predict(img)
        predicted_letter = int2char(np.argmax(prediction[0]))

        # Append the predicted letter to the string
        predicted_string += predicted_letter
    
    return predicted_string

def predict_text(image, model):
    sub_imgs_0, sub_imgs_1 = split_texts(image)
    key = predict_each(sub_imgs_0, model)
    value = predict_each(sub_imgs_1, model)
    return key, value

def identify_clue(image, model):
    # Return this black image if no clues found
    height = 400
    width = 600
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    ## Color thresholding
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    uh = 130
    us = 255
    uv = 255
    lh = 118
    ls = 103
    lv = 85
    lower_hsv = np.array([lh,ls,lv])
    upper_hsv = np.array([uh,us,uv])

    # Threshold the HSV image to get only blue colors
    binary_image = cv2.inRange(hsv, lower_hsv, upper_hsv)

    ## Find contour
    # Find Canny edges 
    edged = cv2.Canny(binary_image, 30, 200)
    contours, hierarchy = cv2.findContours(edged,  
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if len(contours) == 0:
        # print('No Contours detected\n')
        return black_image, None, None

    image_copy = image.copy()

    board_cnt = []
    for cnt in contours:
        if (len(cnt) > len(board_cnt)):
            board_cnt = cnt
            # cv2.drawContours(image_copy, board_cnt, -1, (0, 255, 0), 3)
    if len(board_cnt) <= 500:
        # print('Contour not big enough\n')
        return black_image, None, None

    ## Find edges in contour
    epsilon = 0.05 * cv2.arcLength(board_cnt, True)
    approx_corners = cv2.approxPolyDP(board_cnt, epsilon, True)

    # Check if we have four corners
    if len(approx_corners) == 4:
        # Reshape for convenience
        points = approx_corners.reshape(4, 2)

        # Compute the centroid
        centroid = np.mean(points, axis=0)

        # Sort the points based on their relation to the centroid
        top = points[np.where(points[:, 1] < centroid[1])]
        bottom = points[np.where(points[:, 1] >= centroid[1])]

        sorted_points = np.zeros((4, 2), dtype=np.float32)

        sorted_points[0] = top[np.argmin(top[:, 0])]  # Top-left point has the smallest x value
        sorted_points[1] = top[np.argmax(top[:, 0])]  # Top-right point has the largest x value
        sorted_points[2] = bottom[np.argmax(bottom[:, 0])]  # Bottom-right point has the largest x value
        sorted_points[3] = bottom[np.argmin(bottom[:, 0])]  # Bottom-left point has the smallest x value

        # sorted_points now contains the corners in the order: top-left, top-right, bottom-right, bottom-left
    else:
        # print('Contour is not a rectangle\n')
        return black_image, None, None

    top_left = sorted_points[0]
    top_right = sorted_points[1]
    bottom_right = sorted_points[2]
    bottom_left = sorted_points[3]

    # Subtract 10 pixels in both x and y directions to avoid unwanted corners
    # top_left = (top_left[0] + 5, top_left[1] + 5)
    # top_right = (top_right[0] - 5, top_right[1] + 5)
    # bottom_right = (bottom_right[0] - 5, bottom_right[1] - 5)
    # bottom_left = (bottom_left[0] + 5, bottom_left[1] - 5)

    ## First perspective transform
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    width, height = 600, 400
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(image, M, (width, height))

    ## Locate corners again
    # Apply Harris Corner Detector
    block_size = 5
    aperture_size = 5
    k = 0.08
    harris_corners = cv2.cornerHarris(warped_image[:, :, 2], block_size, aperture_size, k)

    # Dilate the corner points to enhance them
    harris_corners = cv2.dilate(harris_corners, None)

    # Threshold for an optimal value, identify strong corners
    threshold = 0.01 * harris_corners.max()

    corners = []

    # Draw solid red circles on each corner
    for y in range(harris_corners.shape[0]):
        for x in range(harris_corners.shape[1]):
            if harris_corners[y, x] > threshold:
                # Eliminate bad corners lcoated close to the edges of the image
                cond_0 = x >= 10 and x <= 590
                cond_1 = y >= 10 and y <= 390
                if cond_0 and cond_1:
                    corners.append((x, y))
    if (len(corners) < 4):
        # print('Not enough corners in harris corners\n')
        return black_image, None, None
    
    # for x, y in corners:
    #     cv2.circle(warped_image, (x, y), 10, (255, 0, 0), -1)
    # return warped_image

    top_left, top_right, bottom_right, bottom_left = find_corners(corners)

    # def isRect(top_left, top_right, bottom_right, bottom_left):
    #     cx = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
    #     cy = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4

    #     dd1 = np.sqrt(np.abs(cx - top_left[0])) + np.sqrt(np.abs(cy - top_left[1]))
    #     dd2 = np.sqrt(np.abs(cx - top_right[0])) + np.sqrt(np.abs(cy - top_right[1]))
    #     dd3 = np.sqrt(np.abs(cx - bottom_right[0])) + np.sqrt(np.abs(cy - bottom_right[1]))
    #     dd4 = np.sqrt(np.abs(cx - bottom_left[0])) + np.sqrt(np.abs(cy - bottom_left[1]))
        
    #     cond_0 = np.abs(dd1 - dd2) <= 5
    #     cond_1 = np.abs(dd1 - dd3) <= 5
    #     cond_2 = np.abs(dd1 - dd4) <= 5

    #     return cond_0 and cond_1 and cond_2

    # if (is_rectangle(top_left, top_right, bottom_right, bottom_left) == False):
    #     print('Clue board is not a rectangle\n')
    #     return black_image

    for x, y in top_left, top_right, bottom_right, bottom_left:
        # print((x, y))
        cv2.circle(warped_image, (x, y), 10, (255, 0, 0), -1)
    
    ## Second perspective transform
    src_pts = np.float32([top_left, top_right, bottom_right, bottom_left])
    width, height = 600, 400
    dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation
    result = cv2.warpPerspective(warped_image, M, (width, height))

    predicted_key, predicted_val = predict_text(result, model)
    return result, predicted_key, predicted_val
