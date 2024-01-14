import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取视频文件
video_path = 'test3.mp4'
cap = cv2.VideoCapture(video_path)

# 定义特征点跟踪器
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# 初始化特征点
prev_frame = None
prev_pts = None

frame_count = 0  # 记录帧数

# 存储每个特征点的速度
speeds = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # 如果是第一帧，则初始化特征点
    if prev_frame is None:
        prev_frame = frame
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, (36, 25, 25), (86, 255, 255))
        prev_pts = cv2.goodFeaturesToTrack(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY), mask=mask, **feature_params)
        continue

    # 创建空白的掩码图像
    mask = np.zeros_like(frame)

    # 计算光流并提取位移
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                                                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                                    prev_pts, None, **lk_params)

    # 通过状态(status)来过滤掉没有位移的节点
    good_pts = curr_pts[status == 1]
    good_prev_pts = prev_pts[status == 1]

    # 如果没有特征点，则跳过本次循环
    if len(good_pts) == 0:
        prev_frame = frame.copy()
        prev_pts = curr_pts
        continue

    # 计算速度向量的模长作为速度值
    speed = np.linalg.norm(good_pts - good_prev_pts, axis=1)

    # 将速度值添加到speeds列表中
    speeds.append(speed)

    # 绘制光流场和特征点
    for i, (new, old) in enumerate(zip(good_pts, good_prev_pts)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 3, (0, 255, 0), -1)

    img = cv2.add(frame, mask)

    # 显示图像
    cv2.imshow('Optical Flow', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # 更新上一帧的数据
    prev_frame = frame.copy()
    prev_pts = curr_pts

    frame_count += 1

# 将速度列表转换为二维数组
speeds = np.concatenate(speeds)

# 绘制每个特征点的速度曲线
for i in range(speeds.shape[0]):
    plt.plot(speeds[i])

plt.xlabel('Frame')
plt.ylabel('Speed')
plt.title('Speed of Feature Points')
plt.show()

cap.release()
cv2.destroyAllWindows()
