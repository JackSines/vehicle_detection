import sys

import numpy
import torch
import cv2 as cv
import csv
from matplotlib import pyplot as plt
from tkinter import _tkinter
import numpy as np
import yolov5.detect
import glob
from os.path import exists
import os
import time
from PIL import Image
import math
import PySimpleGUI as gui
#from yolov5.detect import run


class detected:
    def __init__(self, x1, y1, x2, y2, confidence, classification):
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.confidence = confidence
        self.classification = classification

        # coordinates in format x1,y1, x2,y2, confidence, class
        # from top left corner

        # Classifications:
        # 0: pedestrian
        # 1: rider
        # 2: car
        # 3: truck
        # 4: bus
        # 5: train
        # 6: motorcycle
        # 7: bicycle
        # 8: traffic light
        # 9: traffic sign


def get_objects(filepath, cpu):
    print('\nStarting object detection...')
    # return bounding box coordinates, computing time, image with bounding box drawn

    #can run object detector with cpu in case of issues
    if cpu:
        bb, t, img_bb = yolov5.detect.run('../yolov5/runs/train/exp3/weights/best.pt', filepath, device='cpu',
                                          imgsz=(256, 416))
    else:
        #this will auto detect what to use - prioritising gpu
         bb, t, img_bb = yolov5.detect.run('../yolov5/runs/train/exp3/weights/best.pt', filepath,imgsz=(256, 416))
    object_list = []
    classes = dict(
        [(0, 'pedestrian'), (1, 'rider'), (2, 'car'), (3, 'truck'), (4, 'bus'), (5, 'train'), (6, 'motorcycle'),
         (7, 'bicycle'), (8, 'traffic light'), (9, 'traffic sign')])

    # the following loop reformats the data from the yolo object detector into x1,y1,x2,y1,confidence, classification
    # each object detected then has a 'detected' object created with this data
    for i in bb:
        car = (str(i)).replace('tensor([', '')
        if "cuda" in car:
            car = car.replace('device=\'cuda:0\')', '')
        else:
            car = car.replace(')', '')
        car = car.replace(']', '')
        car = car.split(', ')
        x1 = car[0]
        y1 = car[1]
        x2 = car[2]
        y2 = car[3]
        confidence = car[4]
        classification = classes[int(float(car[5]))]  # int(float(car[5]))
        obj = detected(x1, y1, x2, y2, confidence, classification)
        object_list.append(obj)
    print('Object detection complete.')
    # write image with bounding boxes to temp file
    cv.imwrite('./temp/detected.png', img_bb)
    return object_list, t, img_bb


def get_vehicles(object_list):
    vehicles = []
    for i in object_list:
        if 1 <= i.classification <= 6 and i.classification != 5:
            # every vehicle - leaving out bicycle as react differently to them
            vehicles.append(i)

    return vehicles
    print('\nVehicles returned.')


def get_cam_details(L, S):
    # if L=true calibrating left camera
    # if S true, calibrating a single camera
    # if neither, calibrating right
    # this decides the name the calibration data is saved under

    if S:
        filepath_matrix = './calibration_data/camera_matrix.csv'
        filepath_distortion = './calibration_data/camera_distortion.csv'
        images = glob.glob('./calibration/mono/*.png')
        filepath_rotation = './calibration_data/camera_rotation.csv'
        filepath_translation = './calibration_data/camera_translation.csv'
    else:
        if L:
            filepath_matrix = './calibration_data/camera_matrix_L.csv'
            filepath_distortion = './calibration_data/camera_distortion_L.csv'
            filepath_rotation = './calibration_data/camera_rotation_L.csv'
            filepath_translation = './calibration_data/camera_translation_L.csv'
            images = glob.glob('./calibration/left/*.png')
        else:
            filepath_matrix = './calibration_data/camera_matrix_R.csv'
            filepath_distortion = './calibration_data/camera_distortion_R.csv'
            filepath_rotation = './calibration_data/camera_rotation_R.csv'
            filepath_translation = './calibration_data/camera_translation_R.csv'
            images = glob.glob('./calibration/right/*.png')

    print('\nCalibrating camera...')

    # create arrays to store object (3d) points and image (2d) points
    world_points = []  # 3d
    img_points = []  # 2d
    board = (9, 6)  # number of squares
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # 3d points - checkerboard square size = 24mm
    square_size = 24
    board_mm = (board[0] * square_size, board[1] * square_size)

    # populate world points and object points by iterating through images
    obj = np.zeros((1, board[0] * board[1], 3), np.float32)
    obj[0, :, :2] = np.mgrid[0:board_mm[0]:square_size, 0:board_mm[1]:square_size].T.reshape(-1, 2)
    w = 0
    h = 0

    for i in images:
        img = cv.imread(i)
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_size = img_grey.shape[::-1]
        # find corners
        found, corners = cv.findChessboardCorners(img_grey, board, cv.CALIB_CB_ADAPTIVE_THRESH
                                                  + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            # if chessboard is found, update world points and calculate image points
            world_points.append(obj)
            corners_ref = cv.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

            img_points.append(corners_ref)
        else:
            # if no board found, image not used
            print("no board found")
            # print(i)

        h, w = img_grey.shape[:2]

    size = (w, h)

    returned, matrix, distortion, rotation, translation = cv.calibrateCamera(world_points, img_points, size, None, None)

    # write to file
    np.savetxt(filepath_matrix, matrix, delimiter=',')
    np.savetxt(filepath_distortion, distortion, delimiter=',')
    # np.save(filepath_rotation, rotation)
    # np.savetxt(filepath_translation, translation, delimiter=',')

    print('Calibration data saved.')


def get_cam_details_stereo():
    # this method is a similar process to get_cam_details()
    print('\nCalibrating stereo cameras...')
    # find images in folder
    images_L = glob.glob('./calibration/left_stereo/*.png')
    images_R = glob.glob('./calibration/right_stereo/*.png')

    # create arrays to store object (3d) points and image (2d) points
    world_points = []  # 3d
    img_points_L = []  # 2d
    img_points_R = []  # 2d
    board = (9, 6)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.01)  # change this

    # 3d points - checkerboard squares = 24mm
    square_size = 24
    board_mm = (board[0] * square_size, board[1] * square_size)

    obj = np.zeros((1, board[0] * board[1], 3), np.float32)  # create empty grid
    obj[0, :, :2] = np.mgrid[0:board_mm[0]:square_size, 0:board_mm[1]:square_size].T.reshape(-1, 2)
    w = 0
    h = 0
    img_shape = 0
    # loop over images
    for i in images_L:
        print('looping left images...')
        img = cv.imread(i)
        # convert to greyscale
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = img_grey.shape[::-1]
        # find corners
        found, corners = cv.findChessboardCorners(img_grey, board, cv.CALIB_CB_ADAPTIVE_THRESH
                                                  + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
        # if found == True, refine then add 3d and 2d points to img
        if found:
            print('board found')
            world_points.append(obj)
            corners_ref = cv.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

            img_points_L.append(corners_ref)
        else:
            print("no board found")
            print(i)

        h, w = img_grey.shape[:2]

    for i in images_R:
        print('looping right images...')
        img = cv.imread(i)
        # convert to greyscale
        img_grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = img_grey.shape[::-1]
        # find corners
        found, corners = cv.findChessboardCorners(img_grey, board, cv.CALIB_CB_ADAPTIVE_THRESH
                                                  + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            print('right board found')
            corners_ref = cv.cornerSubPix(img_grey, corners, (11, 11), (-1, -1), criteria)

            img_points_R.append(corners_ref)
            # draw on image
            cv.drawChessboardCorners(img, board, corners_ref, found)
        else:
            print("no right board found")
            print(i)

    size = (w, h)

    # get camera details
    matrix_L = np.loadtxt('./calibration_data/camera_matrix_L.csv', delimiter=',')  # matrix_L
    matrix_R = np.loadtxt('./calibration_data/camera_matrix_R.csv', delimiter=',')  # matrix_R
    distortion_L = np.loadtxt('./calibration_data/camera_distortion_L.csv', delimiter=',')  # distortion_L
    distortion_R = np.loadtxt('./calibration_data/camera_distortion_R.csv', delimiter=',')  # distortion_R
    # roi_L = np.loadtxt('./calibration_data/roi_L.csv', delimiter=',')  # roi_L
    # roi_R = np.loadtxt('./calibration_data/roi_R.csv', delimiter=',')  # roi_R
    # new_matrix_L = np.loadtxt('./calibration_data/camera_matrix_new_L.csv', delimiter=',')  # new matrix_L
    # new_matrix_R = np.loadtxt('./calibration_data/camera_matrix_new_R.csv', delimiter=',')  # new matrix_R

    #   rotation_L = np.loadtxt('./calibration_data/rotation_L.csv', delimiter=',')  # rotation_L
    #  rotation_R = np.loadtxt('./calibration_data/rotation_R.csv', delimiter=',')  # rotation_R
    #  translation_L = np.loadtxt('./calibration_data/translation_L.csv', delimiter=',')  # translation_L
    #   translation_R = np.loadtxt('./calibration_data/translation_R.csv', delimiter=',')  # translation_R
    #  returned_L, matrix_L, distortion_L, rotation_L, translation_L = cv.calibrateCamera(world_points, img_points_L,
    #                                                                                    size, None, None)
    #  returned_R, matrix_R, distortion_R, rotation_R, translation_R = cv.calibrateCamera(world_points, img_points_R,
    #                                                                                     size, None, None)

    new_matrix_L, roi_L = cv.getOptimalNewCameraMatrix(matrix_L, distortion_L, size, 1, size)
    new_matrix_R, roi_R = cv.getOptimalNewCameraMatrix(matrix_R, distortion_R, size, 1, size)
    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    criteria = cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001

    retS, new_matrix_L, distortion_L, new_matrix_R, distortion_R, rotation, translation, \
    eMat, fMat = cv.stereoCalibrate(world_points, img_points_L, img_points_R, new_matrix_L, distortion_L, new_matrix_R,
                                    distortion_R, img_shape, criteria, flags)

    # write to file
    np.savetxt('./calibration_data/camera_stereo_distortion_L.csv', distortion_L, delimiter=',')
    np.savetxt('./calibration_data/camera_matrix_new_L.csv', new_matrix_L, delimiter=',')

    np.savetxt('./calibration_data/camera_stereo_distortion_R.csv', distortion_R, delimiter=',')
    np.savetxt('./calibration_data/camera_matrix_new_R.csv', new_matrix_R, delimiter=',')

    # np.save('./calibration_data/world_points', world_points)

    # np.save('./calibration_data/img_points_L', img_points_L)
    # np.save('./calibration_data/img_points_R', img_points_R)

    np.savetxt('./calibration_data/roi_L.csv', roi_L, delimiter=',')
    np.savetxt('./calibration_data/roi_R.csv', roi_R, delimiter=',')

    # np.savetxt('./calibration_data/size.csv', size, delimiter=',')
    # np.savetxt('./calibration_data/shape.csv', img_shape, delimiter=',')

    np.savetxt('./calibration_data/stereo_rotation.csv', rotation, delimiter=',')
    np.savetxt('./calibration_data/stereo_translation.csv', translation, delimiter=',')

    print('Calibration data saved.')


def undistort(img, L, S):
    # if L = True, then processing left image
    # if L  = False, then processing right
    # UNLESS S = True, then processing mono img
    print('\nUndistorting...')
    if S:
        filepath_matrix = './calibration_data/camera_matrix.csv'

        filepath_distortion = './calibration_data/camera_distortion.csv'
        filepath_save = './temp/undistorted_img.jpg'
        filepath_save_matrix = './calibration_data/camera_matrix_new.csv'
        filepath_roi = './calibration_data/roi.csv'
    else:
        if L:
            filepath_matrix = './calibration_data/camera_matrix_L.csv'
            filepath_new_matrix = './calibration_data/camera_matrix_new_L.csv'
            filepath_distortion = './calibration_data/camera_distortion_L.csv'
            filepath_save = './temp/undistorted_img_L.jpg'
            filepath_save_matrix = './calibration_data/camera_matrix_new_L.csv'
            filepath_roi = './calibration_data/roi_L.csv'
        else:
            filepath_matrix = './calibration_data/camera_matrix_R.csv'
            filepath_new_matrix = './calibration_data/camera_matrix_new_R.csv'
            filepath_distortion = './calibration_data/camera_distortion_R.csv'
            filepath_save = './temp/undistorted_img_R.jpg'
            filepath_save_matrix = './calibration_data/camera_matrix_new_R.csv'
            filepath_roi = './calibration_data/roi_R.csv'

    # check camera(s) calibrated
    if exists(filepath_matrix) and exists(filepath_distortion):
        matrix = np.loadtxt(filepath_matrix, delimiter=',')
        distortion = np.loadtxt(filepath_distortion, delimiter=',')
    else:
        print("Calibration files missing - please calibrate camera")
        return

    h, w = img.shape[:2]
    size = (w, h)
    # if single camera, calculate new matrix and roi as it hasn't been calculated already
    if S:
        new_matrix, roi = cv.getOptimalNewCameraMatrix(matrix, distortion, size, 1, size)
    # if one of the stereo cameras, load new matrix and roi as they've already been calculated and saved
    else:
        new_matrix = np.loadtxt(filepath_new_matrix, delimiter=',')
        roi = np.loadtxt(filepath_roi, delimiter=',')

    img_undistorted = cv.undistort(img, matrix, distortion, None, new_matrix)
    x, y, w, h = roi
    img_undistorted_cropped = img_undistorted[int(y):int(y) + int(h), int(x):int(x) + int(w)]
    cv.imwrite(filepath_save, img_undistorted_cropped)
    # save undistorted image to temp file for further processing
    print('Image undistorted and saved.')
    return filepath_save


def rectify_stereo(left_img_path, right_img_path):
    left_img = cv.imread(left_img_path)
    right_img = cv.imread(right_img_path)

    imgl_grey = cv.cvtColor(left_img, cv.COLOR_BGR2GRAY)
    img_shape = imgl_grey.shape[::-1]

    # intrinsics and extrinsics loaded from previous steps
    matrix_L = np.loadtxt('./calibration_data/camera_matrix_new_L.csv', delimiter=',')
    distortion_L = np.loadtxt('./calibration_data/camera_stereo_distortion_L.csv', delimiter=',')
    matrix_R = np.loadtxt('./calibration_data/camera_matrix_new_R.csv', delimiter=',')
    distortion_R = np.loadtxt('./calibration_data/camera_stereo_distortion_R.csv', delimiter=',')
    rotation = np.loadtxt('./calibration_data/stereo_rotation.csv', delimiter=',')
    translation = np.loadtxt('./calibration_data/stereo_translation.csv', delimiter=',')

    # stereoRectify - rotates the images so the they're in the same plane - also returns projection matrices
    rectify_scale = 1
    rect_l, rect_r, proj_mat_l, proj_mat_r, Q, roiL, roiR = cv.stereoRectify(matrix_L, distortion_L,
                                                                             matrix_R, distortion_R, img_shape,
                                                                             rotation, translation, rectify_scale,
                                                                             (0, 0))

    # undistortion
    left_map = cv.initUndistortRectifyMap(matrix_L, distortion_L, rect_l, proj_mat_l, img_shape, cv.CV_16SC2, )
    right_map = cv.initUndistortRectifyMap(matrix_R, distortion_R, rect_r, proj_mat_r, img_shape, cv.CV_16SC2)

    left_rectified = cv.remap(left_img, left_map[0], left_map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    cv.imwrite('./temp/left_rectified.png', left_rectified)
    right_rectified = cv.remap(right_img, right_map[0], right_map[1], cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
    cv.imwrite('./temp/right_rectified.png', right_rectified)
    return left_rectified, right_rectified


def ORB(img_l, img_r):
    orb = cv.ORB_create(nfeatures=1000)
    img_l_gr = cv.cvtColor(img_l, cv.COLOR_BGR2GRAY)
    kpl, descl = orb.detectAndCompute(img_l_gr, None)

    img_r_gr = cv.cvtColor(img_r, cv.COLOR_BGR2GRAY)
    kpr, descr = orb.detectAndCompute(img_r_gr, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING2, crossCheck=True)

    matches = bf.match(descl, descr)
    matches = sorted(matches, key=lambda x: x.distance)

    # if less than 5 matches, do not count as a match
    if len(matches) < 5:
        # returns a disparity value thats too high to be used by the rest of the application
        return 999, 999

    # take the top 150 matches
    top_matches = matches[:150]

    img1_matches = []
    img2_matches = []
    disp = 0
    final_result = 0
    # calculate average displacement in the x axis for the matched points
    # calculate the average match distance - this is used to check on average how accurate the top matches are
    for i in top_matches:
        img1_id = i.queryIdx
        img2_id = i.trainIdx

        x1, y1 = kpl[img1_id].pt
        x2, y2 = kpr[img2_id].pt
        disp = disp + (abs(x2 - x1))
        final_result += i.distance
        # print(disp, final_result)

        # if the disparity in the y axis is smaller than 10, count the points in the average
        # this reduces the chance of incorrect matches
        # (matching features should have a similar y value due to the stereo calibration)
        if (abs(y2 - y1)) <= 10.0:
            img1_matches.append((x1, y1))
            img2_matches.append((x2, y2))

    # if there are matches, calculate average
    if len(img1_matches) != 0:
        disp = disp / len(img1_matches)
    # else, return a high disparity value
    else:
        return 999, 999

    final_result = final_result / len(top_matches)

    return final_result, disp


def get_distance(disp, b):
    # load focal length in pixels
    print("disparity: ", disp)
    matrix_l = np.loadtxt('./calibration_data/camera_matrix_new_L.csv', delimiter=',')
    fl_l = matrix_l[0][0]
    # calculate depth using (baseline * focal length)/disparity - this is rounded to 2dp
    depth = round(((b * fl_l) / float(disp)), 2)
    print("Distance: ", depth)

    if depth <= 500:
        print("Distance less than 500cm")

    return depth


def openingScreen():
    gui.theme('Default1')
    layout = [[gui.Text("Object detection and distance estimation",
                        font=(gui.DEFAULT_FONT, 20, 'underline'))],
              [gui.Text('This application allows you to process images/videos and see the results of object '
                        'detection and distance estimation. Please note that stereo images/videos are needed'
                        ' for distance estimation.', size=(60, 5), font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Text('What type of data do you have?', font=(gui.DEFAULT_FONT, 12, 'underline'))],
              [gui.Radio('Stereo (left and right)', "DATA", key='stereo', default=True, font=(gui.DEFAULT_FONT, 12)),
               gui.Radio('Mono (from a single perspective)', "DATA", key='mono', font=(gui.DEFAULT_FONT, 12))],
              [gui.Text('Do you have videos or images?', font=(gui.DEFAULT_FONT, 12, 'underline'))],
              [gui.Radio('Video(s)', "VID/IMAGE", key='video', default=True,
                         font=(gui.DEFAULT_FONT, 12)),
               gui.Radio('Image(s)', "VID/IMAGE", key='image',
                         font=(gui.DEFAULT_FONT, 12))],
              [gui.Button('OK')],
              [gui.Text('If you dont have access to the camera that took the photos (for example, they are taken '
                        'from the internet), then click below to use object detection without camera calibration.'
                        , font=(gui.DEFAULT_FONT, 12))],
              [gui.Button('No calibration')],
              [gui.Button('Information', pad=((0, 500), 150)), gui.Button('Recalibrate', pad=((500, 0), 150))]]
    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        # No calibration button pressed
        if event == 'No calibration':
            window.close()
            monoNoCalib()

        if event == 'Information':
            window.close()
            information()

        if event == 'Recalibrate':
            window.close()
            calibration()

        # OK button pressed
        if event == 'OK':
            # if mono radio button selected
            if values['mono'] == True:
                # checking if camera calibrated
                if exists('./calibration_data/camera_matrix.csv'):
                    window.close()
                    mono()
                # if not calibrated, open calibration window
                else:
                    print('mono camera not calibrated!!')
                    window.close()
                    calibration()
            # if stereo radio button selected
            if values['stereo'] == True:
                # check if camera calibrated
                if exists('./calibration_data/camera_matrix_L.csv') and exists(
                        './calibration_data/camera_matrix_new_L.csv'):
                    # if video radio button pressed
                    if values['video'] == True:
                        window.close()
                        stereoVideo()
                    # if image radio button pressed
                    elif values['image'] == True:
                        # go to stereo image
                        print('go to stereo image')
                        window.close()
                        stereoImg()
                # open calibration window if not calibrated
                else:
                    print('Stereo cameras not calibrated!!!')
                    window.close()
                    calibration()

        if event == gui.WINDOW_CLOSED:
            break

    window.close()


def draw(l_list, r_list, l_img, r_img, dist_list):
    imgl = l_img
    imgr = r_img
    # loop through objects adding bounding boxes, object name, and distance
    for i in range(len(l_list)):
        imgl = cv.rectangle(l_img, (int(l_list[i].x1), int(l_list[i].y1)), (int(l_list[i].x2), int(l_list[i].y2)),
                            (0, 0, 255),
                            2)
        imgl = cv.rectangle(l_img, (int(l_list[i].x1), int(l_list[i].y1)),
                            (int(l_list[i].x2), (int(l_list[i].y1) - 20)),
                            (0, 0, 0),
                            -1)
        imgl = cv.rectangle(l_img, (int(l_list[i].x1), int(l_list[i].y2)),
                            (int(l_list[i].x2), (int(l_list[i].y2) + 20)),
                            (0, 0, 0),
                            -1)
        cv.putText(imgl, 'Object {}'.format(i), (int(l_list[i].x1), int(l_list[i].y1) - 3), cv.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (255, 255, 255), 2, cv.LINE_AA)

        imgr = cv.rectangle(r_img, (int(r_list[i].x1), int(r_list[i].y1)), (int(r_list[i].x2), int(r_list[i].y2)),
                            (0, 0, 255),
                            2)
        imgr = cv.rectangle(r_img, (int(r_list[i].x1), int(r_list[i].y1)),
                            (int(r_list[i].x2), (int(r_list[i].y1) - 20)),
                            (0, 0, 0),
                            -1)
        imgr = cv.rectangle(r_img, (int(r_list[i].x1), int(r_list[i].y2)),
                            (int(r_list[i].x2), (int(r_list[i].y2) + 20)),
                            (0, 0, 0),
                            -1)
        cv.putText(imgr, 'Object {}'.format(i), (int(r_list[i].x1), int(r_list[i].y1) - 3), cv.FONT_HERSHEY_SIMPLEX,
                   0.6,
                   (255, 255, 255), 2, cv.LINE_AA)
        if len(dist_list) > 0:
            cv.putText(imgr, str(dist_list[i]), (int(r_list[i].x1), int(r_list[i].y2) + 15), cv.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255), 1, cv.LINE_AA)

            cv.putText(imgl, str(dist_list[i]), (int(l_list[i].x1), int(l_list[i].y2) + 15), cv.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255), 1, cv.LINE_AA)

    cv.imwrite('./temp/right_complete.png', imgr)
    cv.imwrite('./temp/left_complete.png', imgl)


def information():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Information", font=(gui.DEFAULT_FONT, 20, 'underline'))],
              [gui.Text('This application includes 3 main processes – camera calibration, object detection '
                        'and distance estimation.', font=(gui.DEFAULT_FONT, 12))],
              [gui.Text('Camera calibration', font=(gui.DEFAULT_FONT, 15, 'underline'))],
              [gui.Text('For accurate results, the cameras must be calibrated in order to remove distortion and align '
                        'the y axis of stereo images. If the camera has not been calibrated before image processing, '
                        'you will automatically be taken to the calibration page. If your camera(s) have changed and '
                        'you wish to recalibrate, there is a button on the home screen which will take you to the '
                        'calibration screen.', size=(100, 4), font=(gui.DEFAULT_FONT, 12), justification='c')],
              [gui.Text('Object detection', font=(gui.DEFAULT_FONT, 15, 'underline'))],
              [gui.Text('This application uses YOLOv5s to detect vehicles, traffic signs, etc. The confidence threshold'
                        ' is 0.45 – anything above this will show as a detected object. The detector was trained on '
                        '70,000 images from the Berkeley DeepDrive dataset (https://www.bdd100k.com/)',
                        size=(100, 4), font=(gui.DEFAULT_FONT, 12), justification='c')],
              [gui.Text('Distance estimation', font=(gui.DEFAULT_FONT, 15, 'underline'))],
              [gui.Text('The distance estimation uses the disparity between the left and right stereo images to '
                        'calculate distance using distance =(baseline * focal length) / displacement.',
                        size=(100, 4), font=(gui.DEFAULT_FONT, 12), justification='c')]
              ]

    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home' or event == 'Back':
            window.close()
            openingScreen()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def stereoVideo():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection and distance estimation", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Include distance estimation', default=True, key='dist', font=(gui.DEFAULT_FONT, 12))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Text('Left Video', font=(gui.DEFAULT_FONT, 12)), gui.FileBrowse(key='left'),
               gui.Text('Right Video', font=(gui.DEFAULT_FONT, 12)),
               gui.FileBrowse(key='right')],
              [gui.Button('OK')]]

    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()
        print(event, values)

        if event == 'Home' or event == 'Back':
            window.close()
            openingScreen()

        if event == 'OK' and values['dist'] == True:
            left = values['left']
            right = values['right']
            window.close()
            stereoVideoResults(left, right, True, values['cpu'])

        elif event == 'OK' and values['dist'] == False:
            left = values['left']
            right = values['right']
            window.close()
            stereoVideoResults(left, right, False, values['cpu'])

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def stereoImg():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection and distance estimation", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Include distance estimation', default=True, key='dist', font=(gui.DEFAULT_FONT, 10))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Text('Left Image', font=(gui.DEFAULT_FONT, 10)), gui.FileBrowse(key='left'),
               gui.Text('Right Image', font=(gui.DEFAULT_FONT, 10)),
               gui.FileBrowse(key='right')],
              [gui.Button('OK')]]
    error_layout = [[gui.Text("No objects detected in those images", font=(gui.DEFAULT_FONT, 20))],
                    [gui.Text('Press "Try again" to return and try with different images.',
                              font=(gui.DEFAULT_FONT, 12))],
                    [gui.Button('Try again')]]

    error_filetype = [[gui.Text("Incorrect file type", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('Press "Try again" to return and try with PNG, JPG or JPEG',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('Try again')]]

    error_unknown = [[gui.Text("Error", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('An unknown error occurred. Press OK to reload this window.',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('OK')]]

    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')
    img_ext_list = ['.png', '.jpg', 'jpeg']

    while True:
        try:
            event, values = window.read()
            print(event, values)

            if event == 'Home' or event == 'Back':
                window.close()
                openingScreen()

            if event == 'OK':
                start = time.time()
                left = values['left']
                right = values['right']
                ext_l = os.path.splitext(values['left'])[-1].lower()
                ext_r = os.path.splitext(values['right'])[-1].lower()
                if ext_l in img_ext_list and ext_r in img_ext_list:
                    rect_start = time.time()
                    rectify_stereo(left, right)
                    l_undistort = cv.imread('./temp/left_rectified.png')
                    r_undistort = cv.imread('./temp/right_rectified.png')
                    rect_end = time.time()

                    print("Rectification time: ", rect_end - rect_start)

                    dist = []
                    print('CPU: ', values['cpu'])
                    l_object_list, l_t, l_img_bb = get_objects('./temp/left_rectified.png', values['cpu'])
                    r_object_list, r_t, r_img_bb = get_objects('./temp/right_rectified.png', values['cpu'])
                    l_object_list, r_object_list, disparity_list = checkObjectMatches(l_object_list, r_object_list,
                                                                                      l_undistort, r_undistort)

                    if len(l_object_list) == 0 or len(r_object_list) == 0:
                        window.close()
                        error_window = gui.Window('Error', error_layout, element_justification='c')
                        event, values = error_window.read()
                        if event == 'Try again':
                            error_window.close()
                            stereoImg()

                        if event == gui.WINDOW_CLOSED:
                            break

                    else:
                        match_s = time.time()

                        match_e = time.time()
                        print("Matching time: ", match_e - match_s)
                        i = 0
                        if values['dist']:
                            while i < len(l_object_list) and i < len(r_object_list):
                                dist_test = get_distance(disparity_list[i], 16)
                                dist.append(dist_test)
                                i += 1

                        end = time.time()
                        draw(l_object_list, r_object_list, l_undistort, r_undistort, dist)
                        window.close()

                        print("Time elapsed: ", (end - start))

                        stereoImgResults(l_object_list, dist)

                elif ext_l not in img_ext_list or ext_r not in img_ext_list:
                    window.close()
                    error_window = gui.Window('Error', error_filetype, element_justification='c')
                    event, values = error_window.read()
                    if event == 'Try again':
                        error_window.close()
                        stereoImg()

                    if event == gui.WINDOW_CLOSED:
                        break

            if event == gui.WINDOW_CLOSED:
                break

            window.close()

        except Exception as e:
            print(e)
            window.close()
            error_window = gui.Window('Error', error_unknown, element_justification='c')
            event, values = error_window.read()
            if event == 'OK':
                error_window.close()
                stereoImg()

            if event == gui.WINDOW_CLOSED:
                break
            return


def stereoVideoResults(left, right, dist_show, cpu):
    left_img = './temp/left_complete.png'
    right_img = './temp/right_complete.png'
    table_data = [[]]
    headings = ['Object', 'Classification', 'Distance (cm)', 'Confidence']

    gui.set_options(suppress_error_popups=True)

    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results - stereo", font=(gui.DEFAULT_FONT, 20))],
              [gui.Text("Left", pad=((250, 125), 3)), gui.Text("Right", pad=((125, 250), 3))],
              [gui.Image(left_img, key='left'), gui.Image(right_img, key='right')],
              [gui.Button('Go', key='Go')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50, key='table')]
              ]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home':
            window.close()
            openingScreen()
        if event == 'Back':
            window.close()
            stereoVideo()

        vid_l = cv.VideoCapture(left)
        vid_r = cv.VideoCapture(right)
        print("reading left")
        l_ret, l_frame = vid_l.read()

        print("reading right")
        r_ret, r_frame = vid_r.read()
        try:
            loop = 1
            while l_ret and r_ret and loop==1:
                window.refresh()

                l_frame = cv.resize(l_frame, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
                r_frame = cv.resize(r_frame, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
                cv.imwrite("./temp/l_frame.png", l_frame)
                cv.imwrite("./temp/r_frame.png", r_frame)

                rectify_stereo("./temp/l_frame.png", "./temp/r_frame.png")
                l_undistort = cv.imread('./temp/left_rectified.png')
                r_undistort = cv.imread('./temp/right_rectified.png')
                dist = []
                l_object_list, l_t, l_img_bb = get_objects('./temp/left_rectified.png', cpu)
                r_object_list, r_t, r_img_bb = get_objects('./temp/right_rectified.png', cpu)

                if len(l_object_list) == 0 or len(r_object_list) == 0:
                    window['left'].update('./temp/left_rectified.png')
                    window['right'].update('./temp/right_rectified.png')

                    print("update")
                    l_ret, l_frame = vid_l.read()
                    r_ret, r_frame = vid_r.read()


                elif len(l_object_list) > 0 or len(r_object_list) > 0:
                    l_object_list, r_object_list, disparity_list = checkObjectMatches(l_object_list, r_object_list,
                                                                                      l_undistort, r_undistort)
                    i = 0
                    if dist_show:
                        while i < len(l_object_list) and i < len(r_object_list):
                            distance = get_distance(disparity_list[i], 16)
                            dist.append(distance)
                            i += 1

                        table_data = []
                        for i in range(len(l_object_list)):
                            table_data.append(
                                [i, l_object_list[i].classification, dist[i], l_object_list[i].confidence])

                    if not dist_show:
                        table_data = []
                        for i in range(len(l_object_list)):
                            table_data.append([i, l_object_list[i].classification, 'n/a', l_object_list[i].confidence])

                    draw(l_object_list, r_object_list, l_undistort, r_undistort, dist)

                    window['left'].update('./temp/left_complete.png')
                    window['right'].update('./temp/right_complete.png')
                    # update table
                    window['table'].update(values=table_data)

                    l_ret, l_frame = vid_l.read()
                    r_ret, r_frame = vid_r.read()

        except Exception as e:
            window.close()
            openingScreen()

        if event == gui.WINDOW_CLOSED:
            break
        window.close()


def monoVideoResults(video, cpu):
    img = './temp/left_complete.png'
    table_data = [[]]
    headings = ['Object', 'Classification', 'Confidence']

    gui.set_options(suppress_error_popups=True)

    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results - mono", font=(gui.DEFAULT_FONT, 20))],
              [gui.Image(img, key='img')],
              [gui.Button('Go', key='Go')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50, key='table')]]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        vid = cv.VideoCapture(video)
        print("reading video")
        ret, frame = vid.read()
        try:
            while ret:
                window.refresh()
                frame = cv.resize(frame, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
                print("looping")

                undistort(frame, False, True)

                object_list, img_bb, t = get_objects('./temp/undistorted_img.jpg', cpu)

                table_data = []
                for i in range(len(object_list)):
                    table_data.append([i, object_list[i].classification, object_list[i].confidence])

                window['img'].update('./temp/detected.png')
                # update table
                window['table'].update(values=table_data)

                ret, frame = vid.read()

        except Exception as e:
            window.close()
            openingScreen()

        if event == 'Home':
            window.close()
            openingScreen()
        if event == 'Back':
            window.close()
            mono()

        if event == gui.WINDOW_CLOSED:
            break
        window.close()


def checkObjectMatches(l_obj_list, r_obj_list, img_l, img_r):
    # set list with fewest detections as list 1
    if len(l_obj_list) <= len(r_obj_list):
        list_1 = l_obj_list
        list_2 = r_obj_list
        img_1 = img_l
        img_2 = img_r
        left = True
    else:
        list_1 = r_obj_list
        list_2 = l_obj_list
        img_1 = img_r
        img_2 = img_l
        left = False

    # loop through all combinations of object matches to find closest matches
    new_1 = []
    new_2 = []
    disparity_list = []

    for i in list_1:
        mask = np.zeros(img_1.shape[:2], dtype="uint8")
        cv.rectangle(mask, (int(i.x1), int(i.y1)), (int(i.x2), int(i.y2)), 255, -1)
        img_1_masked = cv.bitwise_and(img_1, img_1, mask=mask)

        best_match = 0
        match_score = 500  # high match score set before loop
        # low match score = closer match
        disparity = 0

        for j in list_2:
            mask2 = np.zeros(img_2.shape[:2], dtype="uint8")

            cv.rectangle(mask2, (int(j.x1), int(j.y1)), (int(j.x2), int(j.y2)), 255, -1)
            img_2_masked = cv.bitwise_and(img_2, img_2, mask=mask2)

            matches, disp = ORB(img_1_masked, img_2_masked)

            # if match score lower than the lowest so far update match score
            if matches < match_score:
                match_score = matches
                best_match = j
                disparity = disp

        # if the lowest match score is less than 27, these 2 objects match
        if match_score <= 27:

            new_1.append(i)
            new_2.append(best_match)
            disparity_list.append(disparity)

        # if higher than this, object from first list discarded
        else:
            print('no objects added')

    if left:
        print('left: ', len(new_1), 'right: ', len(new_2))
        return new_1, new_2, disparity_list

    else:
        print('left: ', len(new_2), 'right: ', len(new_1))
        return new_2, new_1, disparity_list


def calibration():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Camera calibration",
                        font=(gui.DEFAULT_FONT, 20))],
              [gui.Text('What type of camera do you have?', font=(gui.DEFAULT_FONT, 12))],
              [gui.Radio('Stereo (left and right cameras)', "DATA", key='stereo', default=True,
                         font=(gui.DEFAULT_FONT, 12)),
               gui.Radio('Mono (a single camera)', "DATA", key='mono', font=(gui.DEFAULT_FONT, 12))],
              [gui.Button('OK')]]
    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home' or event == 'Back':
            window.close()
            openingScreen()

        if event == 'OK':
            if values['stereo']:
                window.close()
                stereoCalibration()

            elif values['mono']:
                window.close()
                monoCalibration()
            window.close()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def monoCalibration():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Camera calibration - mono", font=(gui.DEFAULT_FONT, 20))],
              [gui.Text('To calibrate your camera, you must use the calibration chessboard and take at least'
                        ' 15 images from varying angles and distances. Please see below for some examples'
                        ' of these images. ',
                        size=(100, 3),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Text('You can find the image of the chessboard in the application files (/application/calibration)'
                        ' - this should be printed onto an A4 page and used horizontally.',
                        size=(100, 2),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Image('./misc/calib.png')],
              [gui.Text("The calibration should be placed in the application folder (./Application/calibration/mono/)",
                        font=(gui.DEFAULT_FONT, 12))],
              [gui.Button('Calibrate')]]
    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home':
            window.close()
            openingScreen()

        if event == 'Back':
            window.close()
            calibration()

        if event == 'Calibrate':
            get_cam_details(False, True)
            window.close()
            openingScreen()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def stereoCalibration():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Camera calibration - stereo", font=(gui.DEFAULT_FONT, 20))],
              [gui.Text('To calibrate your cameras, you must use the calibration chessboard and take at least 15 images'
                        ' from varying angles and distances – this must be done for each camera and then for both '
                        'cameras together. Please see below for some examples of these images. The chessboard should '
                        'be fully visible in both cameras images when taking the images for both cameras together.',
                        size=(100, 3),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Text('You can find the image of the chessboard in the application files (/application/calibration)'
                        ' - this should be printed onto an A4 page and used horizontally. Please note that the stereo '
                        'images should be named so they can be matched up, for example, the first pair of images should'
                        ' be named "left_1", "right_1".',
                        size=(100, 3),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Text('For each individual camera:',
                        size=(100, 1),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Image('./misc/calib.png')],
              [gui.Text('For both cameras at once:',
                        size=(100, 1),
                        font=(gui.DEFAULT_FONT, 12),
                        justification='center')],
              [gui.Image('./misc/stereocalib.png')],
              [gui.Text(
                  "The calibration images should be placed in: ./calibration/ - there is a folder for left, right, "
                  "left stereo and right stereo.",
                  font=(gui.DEFAULT_FONT, 12))],
              [gui.Button('Calibrate')]]
    window = gui.Window('Object Detection and distance estimation', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home':
            window.close()
            openingScreen()

        if event == 'Back':
            window.close()
            calibration()

        if event == 'Calibrate':
            get_cam_details(True, False)
            get_cam_details(False, False)
            get_cam_details_stereo()
            window.close()
            openingScreen()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def stereoImgResults(l_object_list, dist):
    # once images drawn on, they are saved in temp as left/right_complete.png
    # images retrieved from here to be displayed
    table_data = []
    if dist:
        for i in range(len(l_object_list)):
            table_data.append([i, l_object_list[i].classification, dist[i], l_object_list[i].confidence])

        headings = ['Object', 'Classification', 'Distance (cm)', 'Confidence']
    if not dist:
        for i in range(len(l_object_list)):
            table_data.append([i, l_object_list[i].classification, l_object_list[i].confidence])

        headings = ['Object', 'Classification', 'Confidence']

    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results - stereo", font=(gui.DEFAULT_FONT, 20))],
              [gui.Text("Left", font=(gui.DEFAULT_FONT, 15), pad=((250, 125), 0)),
               gui.Text("Right", font=(gui.DEFAULT_FONT, 15), pad=((125, 250), 0))],
              [gui.Image('./temp/left_complete.png', key='left'), gui.Image('./temp/right_complete.png', key='right')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50)]]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        if event == 'Home':
            window.close()
            openingScreen()
        if event == 'Back':
            window.close()
            stereoImg()

        if event == gui.WINDOW_CLOSED:
            break
        window.close()


def mono():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Text('Image/Video', font=(gui.DEFAULT_FONT, 10)), gui.FileBrowse(key='file')],
              [gui.Button('OK')]]

    error_filetype = [[gui.Text("Incorrect file type", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('Press "Try again" to return and try with PNG, JPG, JPEG, MP3, MP4 or AVI',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('Try again')]]

    error_unknown = [[gui.Text("Error", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('An unknown error occurred. Press OK to reload this window.',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('OK')]]

    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')
    while True:
        try:
            event, values = window.read()
            print(event, values)
            if event == 'Home' or event == 'Back':
                window.close()
                openingScreen()

            if event == 'OK':
                file_path = values['file']
                # finds file extension
                ext = os.path.splitext(file_path)[-1].lower()
                # list of supported file extensions for images and video
                img_ext_list = ['.png', '.jpg', 'jpeg']
                video_ext_list = ['.mp3', '.mp4', '.avi']

                if ext in img_ext_list:
                    img = cv.imread(file_path)
                    img_undistorted = undistort(img, False, True)

                    object_list, t, img_bb = get_objects(img_undistorted, values['cpu'])
                    window.close()
                    monoImgResults(object_list)

                elif ext in video_ext_list:
                    window.close()
                    noCalibVideoResults(file_path, values['cpu'])

                elif ext not in img_ext_list or ext not in video_ext_list:
                    print("unsupported file type")
                    window.close()
                    error_window = gui.Window('Error', error_filetype, element_justification='c')
                    event, values = error_window.read()
                    if event == 'Try again':
                        error_window.close()
                        mono()

                    if event == gui.WINDOW_CLOSED:
                        break

            if event == gui.WINDOW_CLOSED:
                break

            window.close()

        except Exception as e:
            print(e)
            window.close()
            error_window = gui.Window('Error', error_unknown, element_justification='c')
            event, values = error_window.read()
            if event == 'OK':
                error_window.close()
                mono()

            if event == gui.WINDOW_CLOSED:
                break
            return


def monoVideo():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Text('Video', font=(gui.DEFAULT_FONT, 12)), gui.FileBrowse(key='video')],
              [gui.Button('OK')]]

    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()
        print(event, values)
        if event == 'Home' or event == 'Back':
            window.close()
            openingScreen()

        if event == 'OK':
            vid_path = values['video']
            window.close()
            monoVideoResults(vid_path, values['cpu'])

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def monoNoCalib():
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection without calibration", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Text('Video/Image', font=(gui.DEFAULT_FONT, 12)), gui.FileBrowse(key='file')],
              [gui.Button('OK')]]

    error_layout = [[gui.Text("No objects detected in those images", font=(gui.DEFAULT_FONT, 20))],
                    [gui.Text('Press "Try again" to return and try with different images.',
                              font=(gui.DEFAULT_FONT, 12))],
                    [gui.Button('Try again')]]

    error_filetype = [[gui.Text("Incorrect file type", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('Press "Try again" to return and try with PNG, JPG, JPEG, MP3, MP4 or AVI',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('Try again')]]

    error_unknown = [[gui.Text("Error", font=(gui.DEFAULT_FONT, 20))],
                      [gui.Text('An unknown error occurred. Press OK to reload this window.',
                                font=(gui.DEFAULT_FONT, 12))],
                      [gui.Button('OK')]]


    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        try:
            event, values = window.read()
            print(event, values)
            if event == 'Home' or event == 'Back':
                window.close()
                openingScreen()

            if event == 'OK':
                file_path = values['file']

                # finds file extension
                ext = os.path.splitext(file_path)[-1].lower()
                # list of supported file extensions for images and video
                img_ext_list = ['.png', '.jpg', 'jpeg']
                video_ext_list = ['.mp3', '.mp4', '.avi']

                if ext in img_ext_list:
                    print('image detected')
                    object_list, t, img_bb = get_objects(file_path, values['cpu'])
                    if len(object_list) == 0:
                        # error
                        window.close()
                        error_window = gui.Window('Error', error_layout, element_justification='c')
                        event, values = error_window.read()
                        if event == 'Try again':
                            error_window.close()
                            monoNoCalib()
                    else:
                        window.close()
                        file = cv.imread('./temp/detected.png')
                        file_resized = cv.resize(file, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
                        cv.imwrite('./temp/detected.png', file_resized)

                        noCalibImgResults(object_list)

                elif ext in video_ext_list:
                    print('video detected')
                    window.close()
                    noCalibVideoResults(file_path)

                elif ext not in img_ext_list or ext not in video_ext_list:
                    print("unsupported file type")
                    window.close()
                    error_window = gui.Window('Error', error_filetype, element_justification='c')
                    event, values = error_window.read()
                    if event == 'Try again':
                        error_window.close()
                        monoNoCalib()

                    if event == gui.WINDOW_CLOSED:
                        break

            if event == gui.WINDOW_CLOSED:
                break

            window.close()

        except Exception as e:
            print(e)
            window.close()
            error_window = gui.Window('Error', error_unknown, element_justification='c')
            event, values = error_window.read()
            if event == 'OK':
                error_window.close()
                stereoImg()

            if event == gui.WINDOW_CLOSED:
                break
            return


def monoImgResults(object_list):
    table_data = []

    for i in range(len(object_list)):
        table_data.append([i, object_list[i].classification, object_list[i].confidence])

    headings = ['Object', 'Classification', 'Confidence']
    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results", font=(gui.DEFAULT_FONT, 20))],
              [gui.Image('./temp/detected.png', key='image')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50)]
              ]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()
        print(event, values)

        if event == 'Home':
            window.close()
            openingScreen()

        if event == 'Back':
            window.close()
            mono()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def noCalibImgResults(object_list):
    table_data = []

    for i in range(len(object_list)):
        table_data.append([i, object_list[i].classification, object_list[i].confidence])

    headings = ['Object', 'Classification', 'Confidence']

    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results", font=(gui.DEFAULT_FONT, 20))],
              [gui.Image('./temp/detected.png', key='image')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50)]]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()
        print(event, values)

        if event == 'Home':
            window.close()
            openingScreen()

        if event == 'Back':
            window.close()
            monoNoCalib()

        if event == gui.WINDOW_CLOSED:
            break

        window.close()


def noCalibVideoResults(video_path, cpu):
    table_data = [[]]
    headings = ['Object', 'Classification', 'Confidence']
    img = './temp/frame.png'

    layout = [[gui.Button('Home'), gui.Button('Back', pad=((5, 1000), 3))],
              [gui.Text("Object detection results - no calibration", font=(gui.DEFAULT_FONT, 20))],
              [gui.Checkbox('Use CPU (please note that performance will be worse with CPU as opposed to GPU',
                            default=False, key='cpu', font=(gui.DEFAULT_FONT, 10))],
              [gui.Image(img, key='img')],
              [gui.Button('Go', key='Go')],
              [gui.Table(values=table_data, headings=headings, max_col_width=50, key='table')]]
    window = gui.Window('Object Detection', layout, size=(1280, 720), element_justification='c')

    while True:
        event, values = window.read()

        vid = cv.VideoCapture(video_path)
        print("reading video")
        ret, frame = vid.read()

        while ret:

            window.refresh()
            frame = cv.resize(frame, (640, 480), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
            print("looping")
            cv.imwrite('./temp/frame.png', frame)

            object_list, img_bb, t = get_objects('./temp/frame.png', cpu)
            table_data = []

            for i in range(len(object_list)):
                table_data.append([i, object_list[i].classification, object_list[i].confidence])

            # update table
            window['table'].update(values=table_data)

            # update image shown
            window['img'].update('./temp/detected.png')
            # read next frame of video
            ret, frame = vid.read()

            if event == 'Home':
                window.close()
                openingScreen()
            if event == 'Back':
                window.close()
                monoNoCalib()

        if event == 'Home':
            window.close()
            openingScreen()
        if event == 'Back':
            window.close()
            monoNoCalib()

        if event == gui.WINDOW_CLOSED:
            break
        window.close()


def main():
    openingScreen()


if __name__ == "__main__":
    main()
