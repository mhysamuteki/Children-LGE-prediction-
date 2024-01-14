import cv2
import os

def get_video_fps(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 关闭视频流
    cap.release()
    
    return fps

def create_video(image_folder, video_path):
    # 获取要合成的图片列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()  # 确保按顺序处理图像

    # 获取第一张图片的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video_path1 = '00743333-4CLV.avi'
    fps = get_video_fps(video_path1)

    # 创建视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用XVID编码器
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # 逐帧写入图像到视频
    for image in images:
        img = cv2.imread(os.path.join(image_folder, image))
        video.write(img)

    # 关闭视频编码器
    video.release()

# 调用函数进行图片合成视频
image_folder = 'output'
video_path = 'video.avi'
create_video(image_folder, video_path)
