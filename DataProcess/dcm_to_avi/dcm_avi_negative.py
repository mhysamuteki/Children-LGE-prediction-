import pydicom
import cv2
import os
from pydicom.pixel_data_handlers.util import convert_color_space

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
        frame_ybr = cv2.cvtColor(frame_8bit, cv2.COLOR_BGR2YCrCb)

        # 将帧写入视频文件
        video_writer.write(frame_8bit)

    video_writer.release()

def convert_folder(folder_path, output_folder):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 构建DICOM文件的完整路径
        dicom_filepath = os.path.join(folder_path, filename)

        # 构建输出AVI文件的路径和名称
        avi_filename = os.path.splitext(filename)[0] + '.avi'
        output_filepath = os.path.join(output_folder, avi_filename)

        # 调用convert_dcm_to_video函数进行转换
        convert_dcm_to_video(dicom_filepath, output_filepath)


input_folder = r'G:\dcm_to_avi\dcm-negative'
output_folder = r'G:\dcm_to_avi\avi-negative'
convert_folder(input_folder, output_folder)

