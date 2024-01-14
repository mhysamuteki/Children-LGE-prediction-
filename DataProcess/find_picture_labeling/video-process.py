import cv2

# 打开视频文件
video = cv2.VideoCapture('4CLV.avi')

# 检查视频文件是否成功打开
if not video.isOpened():
    print("无法打开视频文件")
    exit()

# 读取第一帧
ret, frame = video.read()

# 选择感兴趣区域（ROI）
x, y, w, h = 11, 598, 903, 100  # 设置ROI的左上角坐标和宽度、高度
roi = frame[y:y+h, x:x+w]

# 创建视频编写器
output_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), video.get(cv2.CAP_PROP_FPS), (w, h))

# 处理视频帧
while ret:
    # 写入截取的ROI帧到输出视频文件
    output_video.write(roi)
    
    # 显示截取的ROI帧
    cv2.imshow('ROI', roi)
    
    # 读取下一帧
    ret, frame = video.read()
    
    # 提取ROI帧
    if ret:
        roi = frame[y:y+h, x:x+w]
    
    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放视频对象和关闭窗口
video.release()
output_video.release()
cv2.destroyAllWindows()