import cv2
import numpy as np
import csv

def locate_pixel(video_path, output_csv):
    # 读取视频
    cap = cv2.VideoCapture(video_path)

    # 获取视频的帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 创建一个空的CSV文件
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Time', 'Frame', 'X', 'Y'])

    # 初始化上一帧的坐标和时间戳
    prev_coord = None
    prev_timestamp = None

    # 循环处理每一帧
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 获取当前帧的时间戳
        curr_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # 将图像从BGR颜色空间转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 定义深蓝色的范围（可以根据实际情况调整）
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # 根据颜色范围创建掩膜
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 在掩膜中查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # 获取最大的轮廓
            contour = max(contours, key=cv2.contourArea)

            # 计算轮廓的中心坐标
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # 检查坐标是否发生变化
            if prev_coord is None or (cx, cy) != prev_coord:
                # 将时间戳和坐标写入CSV文件
                csv_writer.writerow([curr_timestamp, curr_timestamp * fps, cx, cy])
                prev_coord = (cx, cy)

        prev_timestamp = curr_timestamp

    # 关闭视频和CSV文件
    cap.release()
    csv_file.close()

# 运行定位函数
video_path = 'output_video.mp4'  # 替换为实际的视频路径
output_csv = 'output2.csv'  # 替换为输出的CSV文件路径
locate_pixel(video_path, output_csv)