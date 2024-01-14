import cv2

def save_frame(video_path, time1, time2, save_path1, save_path2):
    # 打开视频文件
    video = cv2.VideoCapture(video_path)

    # 确保视频文件成功打开
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 将时间转换为视频的帧数
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(frame_rate * time1)
    frame_count2 = int(frame_rate * time2)

    # 设置当前帧数
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    # 读取指定帧
    ret, frame = video.read()

    # 检查帧是否成功读取
    if not ret:
        print("无法读取视频帧")
        return

    # 保存第一个图片
    cv2.imwrite(save_path1, frame)

    # 设置当前帧数为第二个时间点
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count2)

    # 读取指定帧
    ret, frame = video.read()

    # 检查帧是否成功读取
    if not ret:
        print("无法读取视频帧")
        return

    # 保存第二个图片
    cv2.imwrite(save_path2, frame)

    # 释放视频对象
    video.release()

# 示例用法
video_path = "4CLV.avi"
time1 = 0.5333333333333333  # 第一个时间点（秒）
time2 = 3.033333333333333  # 第二个时间点（秒）
save_path1 = "image1.jpg"
save_path2 = "image2.jpg"

save_frame(video_path, time1, time2, save_path1, save_path2)