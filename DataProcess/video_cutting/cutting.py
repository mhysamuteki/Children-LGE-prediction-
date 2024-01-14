import cv2

def extract_frames(video_path, output_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 1

    # 逐帧读取视频并保存为图片
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 保存当前帧为图片
        image_path = f'{output_path}/{frame_count:06d}.jpg'  # 使用完整的文件名
        cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])  # 设置JPEG压缩质量为90，可根据需要调整
        
        frame_count += 1

    # 关闭视频流
    cap.release()

# 调用函数进行视频逐帧分解
video_path = 'test.avi'
output_path = 'output'
extract_frames(video_path, output_path)
