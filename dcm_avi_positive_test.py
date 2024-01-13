import pydicom
import cv2
import numpy as np

def convert_dcm_to_video(dicom_filename, video_name):
    # 读取DICOM文件
    dicom_data = pydicom.dcmread(dicom_filename)

    # 提取和处理图像帧
    frames = dicom_data.pixel_array
    num_frames = frames.shape[0]

    # 将帧转换为视频
    fps = 30  
    height, width = frames.shape[1], frames.shape[2]
    #指定视频编解码器为Xvid
    video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    for frame in frames:
        # 将帧转换为8位无符号整数格式
        frame_8bit = frame.astype('uint8')
        frame_rgb = cv2.cvtColor(frame_8bit, cv2.COLOR_YCrCb2RGB)

        # 将帧写入视频文件
        video_writer.write(frame_rgb)

    video_writer.release()

# 调用函数进行转换
input = '4C-LV'
output = 'output.avi'
convert_dcm_to_video(input, output)