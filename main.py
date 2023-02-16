# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 21:03:38 2021
@author: yangxing
"""

#!/bin/python
import argparse
import glob
# import json
import os
import re

import cv2
import numpy as np
# from tqdm import tqdm

from lxml import etree
import xml.etree.cElementTree as ET

import tkinter as tk
from tkinter import filedialog
import getpass
user_name = getpass.getuser()  # get the user's name
DELAY = 20  # keyboard delay (in milliseconds)
WITH_QT = False
try:
    cv2.namedWindow('Test')
    cv2.displayOverlay('Test', 'Test QT', 500)
    WITH_QT = True
except cv2.error:
    print('-> Please ignore this error message\n')
cv2.destroyAllWindows()


parser = argparse.ArgumentParser(description='Open-source image labeling tool')
parser.add_argument('-i', '--input_dir', default=None,
                    type=str, help='Path to input directory')
parser.add_argument('-o', '--output_dir', default='output_1',# default='output_1'
                    type=str, help='Path to output directory')
parser.add_argument('-t', '--thickness', default='1', type=int,
                    help='Bounding box and cross line thickness')
parser.add_argument('--draw-from-PASCAL-files', action='store_true',
                    help='Draw bounding boxes from the PASCAL files')  # default YOLO
# parser.add_argument('-n', '--n_frames', default='200',
#                     type=int, help='number of frames to track object for')
args = parser.parse_args()

class_index = 0
img_index = 0
img = None
img_objects = []

INPUT_DIR = args.input_dir
OUTPUT_DIR = args.output_dir
# When viewing other people's labeling results, change the user_name in the lower line to the corresponding ID, e.g. 'yangxing'
# merge: outcome of all user's
# camera's position: 'top' or 'lateral'
camPos = 'lateral'
# user_name='Administrator'# user_name = 'merge'
OUTPUT_DIR = os.path.join(OUTPUT_DIR, camPos, user_name)
# OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'merge')
# OUTPUT_DIR = 'D:/data/vidClip/sample/S09-label'
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
# N_FRAMES = args.n_frames
# TRACKER_TYPE = args.tracker


WINDOW_NAME = 'ActionLabeling'
TRACKBAR_IMG = 'Image'
TRACKBAR_CLASS = 'Class'

annotation_formats = {'PASCAL_VOC': '.xml', 'YOLO_darknet': '.txt'}
TRACKER_DIR = os.path.join(OUTPUT_DIR, '.tracker')

DRAW_FROM_PASCAL = args.draw_from_PASCAL_files

# selected bounding box
prev_was_double_click = False
is_bbox_selected = False
selected_bbox = -1
LINE_THICKNESS = args.thickness
POINT_RADIUS = 5

mouse_x = 0
mouse_y = 0
point = (-1, -1)

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
margin = 3
showShortcut = True

# # Check if a point belongs to a rectangle
# def pointInRect(pX, pY, rX_left, rY_top, rX_right, rY_bottom):
#     return rX_left <= pX <= rX_right and rY_top <= pY <= rY_bottom


def display_text(text, time):
    if WITH_QT:
        cv2.displayOverlay(WINDOW_NAME, text, time)
    else:
        print(text)


def set_img_index(x):
    global img_index, img
    img_index = x
    img_path = IMAGE_PATH_LIST[img_index]
    img = cv2.imread(img_path)
    text = 'Showing image {}/{}, path: {}'.format(
        str(img_index), str(last_img_index), img_path)
    display_text(text, 1000)


def set_class_index(x):
    global class_index
    class_index = x
    text = 'Selected class {}/{} -> {}'.format(
        str(class_index), str(last_class_index), CLASS_LIST[class_index])
    display_text(text, 3000)


def draw_edges(tmp_img):
    blur = cv2.bilateralFilter(tmp_img, 3, 75, 75)
    edges = cv2.Canny(blur, 150, 250, 3)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    # Overlap image and edges together
    tmp_img = np.bitwise_or(tmp_img, edges)
    #tmp_img = cv2.addWeighted(tmp_img, 1 - edges_val, edges, edges_val, 0)
    return tmp_img


def decrease_index(current_index, last_index):
    current_index -= 1
    if current_index < 0:
        current_index = last_index
    return current_index


def increase_index(current_index, last_index):
    current_index += 1
    if current_index > last_index:
        current_index = 0
    return current_index


def draw_line(img, x, y, height, width, color):
    cv2.line(img, (x, 0), (x, height), color, LINE_THICKNESS)
    cv2.line(img, (0, y), (width, y), color, LINE_THICKNESS)


def txt_format(class_name, point, width, height):
    # YOLO wants everything normalized
    # Order: class x_center y_center x_width y_height
    x = float(point[0]/width)
    y = float(point[1]/height)
    items = map(str, [class_name, x, y])
    return ' '.join(items)


def voc_format(class_name, point):
    # Order: class_name xmin ymin xmax ymax
    x, y = point
    items = map(str, [class_name, x, y])
    return items


def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)


def append_kp(ann_path, line, extension):
    if '.txt' in extension:
        with open(ann_path, 'a') as myfile:
            myfile.write(line + ' ' + user_name + '\n')  # append line
    elif '.xml' in extension:
        class_name, x, y = line

        tree = ET.parse(ann_path)
        annotation = tree.getroot()

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'user').text = user_name
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        kpoint = ET.SubElement(obj, 'keypoint')
        ET.SubElement(kpoint, 'x').text = x
        ET.SubElement(kpoint, 'y').text = y

        xml_str = ET.tostring(annotation)
        write_xml(xml_str, ann_path)


def yolo_to_voc(x_center, y_center, width, height):
    x_center *= float(width)
    y_center *= float(height)
    return x_center, y_center


def get_xml_object_data(obj):
    class_name = obj.find('name').text
    class_index = CLASS_LIST.index(class_name)
    user = obj.find('user').text
    keypoint = obj.find('keypoint')
    x = int(keypoint.find('x').text)
    y = int(keypoint.find('y').text)
    return [class_name, class_index, x, y, user]


def get_txt_object_data(obj):
    global width, height
    class_name, centerX, centerY, user = obj.split()
    centerX = float(centerX)
    centerY = float(centerY)

    # class_index = int(classId)
    # class_name = CLASS_LIST[class_index]
    class_index = CLASS_LIST.index(class_name)
    x = int(width * centerX)
    y = int(height * centerY)
    return [class_name, class_index, x, y, user]


def draw_kpoints_from_file(tmp_img, annotation_paths):
    global img_objects, font, font_scale  # , is_bbox_selected, selected_bbox
    img_objects = []
    ann_path = None
    if DRAW_FROM_PASCAL:
        # Drawing bounding boxes from the PASCAL files
        ann_path = next(
            path for path in annotation_paths if 'PASCAL_VOC' in path)
    else:
        # Drawing bounding boxes from the YOLO files
        ann_path = next(
            path for path in annotation_paths if 'YOLO_darknet' in path)
    if os.path.isfile(ann_path):
        if DRAW_FROM_PASCAL:
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            for idx, obj in enumerate(annotation.findall('object')):
                class_name, class_index, x, y, user = get_xml_object_data(obj)
                #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                img_objects.append([class_index, x, y])
                # color = class_rgb[class_index].tolist()
                color = class_rgb[-1].tolist()
                # draw point
                cv2.circle(tmp_img, (x, y), POINT_RADIUS,
                           color, LINE_THICKNESS)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(tmp_img, class_name+' / '+user, (x, y - 5),
                            font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        else:
            # Draw from TXT
            with open(ann_path) as fp:
                for idx, line in enumerate(fp):
                    obj = line
                    class_name, class_index, x, y, user = get_txt_object_data(obj)
                    #print('{} {} {} {} {}'.format(class_index, xmin, ymin, xmax, ymax))
                    img_objects.append([class_index, x, y])
                    # color = class_rgb[class_index].tolist()
                    color = class_rgb[-1].tolist()
                    # draw point
                    cv2.circle(tmp_img, (x, y), POINT_RADIUS,
                               color, LINE_THICKNESS)
                    # font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(tmp_img, class_name+' / '+user, (x, y - 5),
                                font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
    return tmp_img


def delete_kpoint(obj_to_edit):

    path = IMAGE_PATH_LIST[img_index]

    class_index, x, y = map(int, obj_to_edit)
    print(obj_to_edit)

    for ann_path in get_annotation_paths(path, annotation_formats):
        if '.txt' in ann_path:
            # edit YOLO file
            with open(ann_path, 'r') as old_file:
                lines = old_file.readlines()

            # yolo_line = txt_format(class_index, (x, y), width, height)
            # ind = findIndex(obj_to_edit)

            with open(ann_path, 'w') as new_file:
                for line in lines:
                    class_name_txt, class_index_txt, x_txt, y_txt, user = get_txt_object_data(
                        line)
                    if class_index_txt != class_index or user != user_name:
                        new_file.write(line)

        elif '.xml' in ann_path:
            # edit PASCAL VOC file
            tree = ET.parse(ann_path)
            annotation = tree.getroot()
            for obj in annotation.findall('object'):
                class_name_xml, class_index_xml, x_xml, y_xml, user = get_xml_object_data(
                    obj)
                if (class_index == class_index_xml and user_name == user):
                    annotation.remove(obj)
                    break

            xml_str = ET.tostring(annotation)
            write_xml(xml_str, ann_path)


def mouse_listener(event, x, y, flags, param):
    # mouse callback function
    global prev_was_double_click, mouse_x, mouse_y, point
    global class_index, last_class_index, img_index, last_img_index

    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y
    # By AlexeyGy: delete via right-click
    elif event == cv2.EVENT_RBUTTONDOWN:
        # selected_kpoint = class_index
        # obj_to_edit = img_objects[selected_kpoint]
        for obj_to_edit in img_objects:
            if obj_to_edit[0] == class_index:
                delete_kpoint(obj_to_edit)

    elif event == cv2.EVENT_LBUTTONDOWN:
        if prev_was_double_click:
            #print('Finish double click')
            prev_was_double_click = False
        else:
            #print('Normal left click')

            if point[0] == -1:
                point = (x, y)

    elif event == cv2.EVENT_MOUSEWHEEL:
        # show previous image wheel listener
        if flags > 0:
           img_index = decrease_index(img_index, last_img_index)
        # show next image wheel listener
        elif flags < 0:
            img_index = increase_index(img_index, last_img_index)
        set_img_index(img_index)
        cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)


def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
    return [int(text) if text.isdigit() else text.lower()
            for text in _nsre.split(s)]


def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def get_annotation_paths(img_path, annotation_formats):
    annotation_paths = []
    for ann_dir, ann_ext in annotation_formats.items():
        new_path = os.path.join(OUTPUT_DIR, ann_dir)
        # img_path.replace(INPUT_DIR, new_path, 1)
        new_path = os.path.join(
            new_path, os.path.basename(os.path.normpath(img_path)))
        pre_path, img_ext = os.path.splitext(new_path)
        new_path = new_path.replace(img_ext, ann_ext, 1)
        annotation_paths.append(new_path)
    return annotation_paths


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


def save_key_points(annotation_paths, class_index, point, width, height):
    for ann_path in annotation_paths:
        if '.txt' in ann_path:
            line = txt_format(CLASS_LIST[class_index], point, width, height)
            append_kp(ann_path, line, '.txt')
        elif '.xml' in ann_path:
            line = voc_format(CLASS_LIST[class_index], point)
            append_kp(ann_path, line, '.xml')


def complement_bgr(color):
    lo = min(color)
    hi = max(color)
    k = lo + hi
    return tuple(k - u for u in color)


def brightnessAndContrastAuto(src):
    img1 = src[:, :, 0]
    img2 = src[:, :, 1]
    img3 = src[:, :, 2]
    clahe = cv2.createCLAHE(3, (8, 8))
    img1 = clahe.apply(img1)
    img2 = clahe.apply(img2)
    img3 = clahe.apply(img3)
    dst = cv2.merge([img1, img2, img3])
    return dst


def read_images(INPUT_DIR):
    global last_img_index, IMAGE_PATH_LIST
    for f in sorted(os.listdir(INPUT_DIR), key=natural_sort_key):
        f_path = os.path.join(INPUT_DIR, f)
        if os.path.isdir(f_path):
            # read images in subdirectories
            # continue
            imglist = glob.glob(os.path.join(f_path,'*.jpg'))
            imglist.sort()
            for imgpath in imglist:
                IMAGE_PATH_LIST.append(imgpath)

        else: # read images in INPUT_DIR         
            test_img = cv2.imread(f_path)
            if test_img is not None:
                IMAGE_PATH_LIST.append(f_path)
    

    last_img_index = len(IMAGE_PATH_LIST) - 1

    # create output directories
    for ann_dir in annotation_formats:
        new_dir = os.path.join(OUTPUT_DIR, ann_dir)
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

    # create empty annotation files for each image, if it doesn't exist already
    for img_path in IMAGE_PATH_LIST:
        # image info for the .xml file
        test_img = cv2.imread(img_path)
        abs_path = os.path.abspath(img_path)
        folder_name = os.path.dirname(img_path)
        image_name = os.path.basename(img_path)
        img_height, img_width, depth = (str(number)
                                        for number in test_img.shape)

        for ann_path in get_annotation_paths(img_path, annotation_formats):
            if not os.path.isfile(ann_path):
                if '.txt' in ann_path:
                    open(ann_path, 'a').close()
                elif '.xml' in ann_path:
                    create_PASCAL_VOC_xml(
                        ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)


# change to the directory of this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    # load all images and videos (with multiple extensions) from a directory using OpenCV
    IMAGE_PATH_LIST = []
    VIDEO_NAME_DICT = {}
    if not INPUT_DIR:
        root = tk.Tk()
        INPUT_DIR = filedialog.askdirectory()
        root.destroy()
        # folderpath = os.path.split(INPUT_DIR)[0]
        # folderpath = os.path.split(folderpath)[0]
        # foldername = os.path.split(INPUT_DIR)[1]
        # OUTPUT_DIR = os.path.join(folderpath,foldername+'-label')
        # if not os.path.exists(OUTPUT_DIR):
        #     os.mkdir(OUTPUT_DIR)
    read_images(INPUT_DIR)
    # for f in sorted(os.listdir(INPUT_DIR), key=natural_sort_key):
    #     f_path = os.path.join(INPUT_DIR, f)
    #     if os.path.isdir(f_path):
    #         # skip directories
    #         continue
    #     # check if it is an image
    #     test_img = cv2.imread(f_path)
    #     if test_img is not None:
    #         IMAGE_PATH_LIST.append(f_path)

    # last_img_index = len(IMAGE_PATH_LIST) - 1

    # # create output directories
    # if len(VIDEO_NAME_DICT) > 0:
    #     if not os.path.exists(TRACKER_DIR):
    #         os.makedirs(TRACKER_DIR)
    # for ann_dir in annotation_formats:
    #     new_dir = os.path.join(OUTPUT_DIR, ann_dir)
    #     if not os.path.exists(new_dir):
    #         os.makedirs(new_dir)
    #     for video_name_ext in VIDEO_NAME_DICT:
    #         new_video_dir = os.path.join(new_dir, video_name_ext)
    #         if not os.path.exists(new_video_dir):
    #             os.makedirs(new_video_dir)

    # # create empty annotation files for each image, if it doesn't exist already
    # for img_path in IMAGE_PATH_LIST:
    #     # image info for the .xml file
    #     test_img = cv2.imread(img_path)
    #     abs_path = os.path.abspath(img_path)
    #     folder_name = os.path.dirname(img_path)
    #     image_name = os.path.basename(img_path)
    #     img_height, img_width, depth = (str(number)
    #                                     for number in test_img.shape)

    #     for ann_path in get_annotation_paths(img_path, annotation_formats):
    #         if not os.path.isfile(ann_path):
    #             if '.txt' in ann_path:
    #                 open(ann_path, 'a').close()
    #             elif '.xml' in ann_path:
    #                 create_PASCAL_VOC_xml(
    #                     ann_path, abs_path, folder_name, image_name, img_height, img_width, depth)

    # load class list
    with open('class_list.txt') as f:
        CLASS_LIST = list(nonblank_lines(f))
    with open('shortcut_list.txt') as f:
        SHORTCUT_LIST = list(nonblank_lines(f))
    # print(CLASS_LIST)
    last_class_index = len(CLASS_LIST) - 1

    # Make the class colors the same each session
    # The colors are in BGR order because we're using OpenCV
    class_rgb = [
        (0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255),
        (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
         (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (0, 0, 0)]
    # (64, 0, 0), (64, 64, 0), (0, 64, 0), (64, 0, 64), (0, 64, 64), (0, 0, 64)]
    class_rgb = np.array(class_rgb)
    # # If there are still more classes, add new colors randomly
    # num_colors_missing = len(CLASS_LIST) - len(class_rgb)
    # if num_colors_missing > 0:
    #     more_colors = np.random.randint(0, 255+1, size=(num_colors_missing, 3))
    #     class_rgb = np.vstack([class_rgb, more_colors])

    # create window
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW_NAME, 1000, 700)
    cv2.setMouseCallback(WINDOW_NAME, mouse_listener)

    # selected image
    cv2.createTrackbar(TRACKBAR_IMG, WINDOW_NAME, 0,
                       last_img_index, set_img_index)

    # selected class
    if last_class_index != 0:
        cv2.createTrackbar(TRACKBAR_CLASS, WINDOW_NAME, 0,
                           last_class_index, set_class_index)

    # initialize
    set_img_index(0)
    edges_on = False

    display_text('Welcome!\n Press [Tab] for help.', 4000)

    # loop
    while True:
        # color = class_rgb[class_index].tolist()
        color = class_rgb[-1].tolist()
        # clone the img
        tmp_img = img.copy()
        tmp_img = brightnessAndContrastAuto(tmp_img)
        height, width = tmp_img.shape[:2]
        if edges_on == True:
            # draw edges
            tmp_img = draw_edges(tmp_img)
        # draw vertical and horizontal guide lines
        # draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
        # write selected class
        class_name = CLASS_LIST[class_index]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 0.6
        # margin = 3
        text_width, text_height = cv2.getTextSize(
            class_name, font, font_scale, LINE_THICKNESS)[0]
        # tmp_img = cv2.rectangle(tmp_img, (mouse_x + LINE_THICKNESS, mouse_y - LINE_THICKNESS), (mouse_x + text_width + margin, mouse_y - text_height - margin), complement_bgr(color), -1)
        tmp_img = cv2.putText(tmp_img, class_name, (mouse_x + margin, mouse_y - margin),
                              font, font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        color = class_rgb[3].tolist()
        if showShortcut:
            for i, name in enumerate(CLASS_LIST):
                tmp_img = cv2.putText(tmp_img, SHORTCUT_LIST[i]+': '+name,
                                      (int(0.8*width) + 5*margin, (10*i+10)*margin), font,
                                      font_scale, color, LINE_THICKNESS, cv2.LINE_AA)
        # get annotation paths
        img_path = IMAGE_PATH_LIST[img_index]
        annotation_paths = get_annotation_paths(img_path, annotation_formats)

        # draw already done bounding boxes
        tmp_img = draw_kpoints_from_file(tmp_img, annotation_paths)

        # if first click
        if point[0] != -1:
            # draw partial bbox
            cv2.circle(tmp_img, point, POINT_RADIUS, color, LINE_THICKNESS)

            # save the bounding box
            save_key_points(annotation_paths, class_index,
                            point, width, height)
            # reset the points
            point = (-1, -1)
            img_index = increase_index(img_index, last_img_index)
            set_img_index(img_index)
            cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)

        cv2.imshow(WINDOW_NAME, tmp_img)
        pressed_key = cv2.waitKey(DELAY)

        if 1:
            ''' Key Listeners START '''
            if pressed_key in range(0x110000):
                if chr(pressed_key) in SHORTCUT_LIST:
                    # point = (np.random.randint(0.25*width, 0.75*width),
                    #          np.random.randint(0.25*height, 0.75*height))
                    point = (mouse_x, mouse_y)
                    class_index = SHORTCUT_LIST.index(chr(pressed_key))
                    cv2.setTrackbarPos(
                        TRACKBAR_CLASS, WINDOW_NAME, class_index)
                elif pressed_key == ord('-') or pressed_key == ord('='):
                    # change down current class key listener
                    if pressed_key == ord('-'):
                        class_index = decrease_index(
                            class_index, last_class_index)
                    # change up current class key listener
                    elif pressed_key == ord('='):
                        class_index = increase_index(
                            class_index, last_class_index)
                    # draw_line(tmp_img, mouse_x, mouse_y, height, width, color)
                    set_class_index(class_index)
                    cv2.setTrackbarPos(
                        TRACKBAR_CLASS, WINDOW_NAME, class_index)

                # help key listener
                elif pressed_key == ord('\t'):  # 'Tab'.
                    text = ('[,] to show edges;\n'
                            '[Esc] to quit;\n'
                            'roll the wheel to change Image;\n'
                            '[-] or [=] to change Class;\n'
                            '[0-9] for shorcut lableling;\n'
                            '[Space] to load next folder.\n'
                            )
                    display_text(text, 5000)
                # show edges key listener
                elif pressed_key == ord(','):  # ','.
                    if edges_on == True:
                        edges_on = False
                        display_text('Edges turned OFF!', 1000)
                    else:
                        edges_on = True
                        display_text('Edges turned ON!', 1000)
                elif pressed_key == ord(' '):  # 'Space'.
                    root = tk.Tk()
                    INPUT_DIR = filedialog.askdirectory()
                    root.destroy()
                    IMAGE_PATH_LIST = []
                    read_images(INPUT_DIR)
                    img_index = 0
                    cv2.setTrackbarPos(TRACKBAR_IMG, WINDOW_NAME, img_index)
                # quit key listener
                elif pressed_key == ord('\r'):  # 'Esc'.
                    showShortcut = not showShortcut
                elif pressed_key == ord('\x1b'):  # 'Esc'.
                    break
                ''' Key Listeners END '''

        if WITH_QT:
            # if window gets closed then quit
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cv2.destroyAllWindows()
