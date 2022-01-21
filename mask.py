
#https://dev.intelrealsense.com/docs/opencv-wrapper#section-2-simple-background-removal-using-the-grabcut-algorithm

import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# ALIGNMENT
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

align_to = rs.stream.color
align = rs.align(align_to)


try:
    while True:

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.convertScaleAbs(depth_image, alpha=1)

        color_image = np.asanyarray(color_frame.get_data())

        #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        ret, x = cv2.threshold(depth_image, 5, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # element = cv2.getStructuringElement(cv2.MORPH_CROSS, (2,2))

        # x = cv2.dilate(x, element)
        # x = cv2.erode(x, element)

        Rcontours, hier_r = cv2.findContours(x, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #A-----------
        print(len(Rcontours))
        cv2.drawContours(color_image, Rcontours, -1, (0, 255, 0), 1)

        #B-----------
        # r_areas = [cv2.contourArea(c) for c in Rcontours]
        # max_rarea = np.max(r_areas)
        # CntExternalMask = np.ones(x.shape[:2], dtype="uint8") * 255

        # for c in Rcontours:
        #     if(( cv2.contourArea(c) > max_rarea * 0.00001) and (cv2.contourArea(c)< max_rarea)):
        #         cv2.drawContours(CntExternalMask,[c],-1,0,1)


        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)

finally:

    # Stop streaming
    pipeline.stop()