import pyautogui
from pynput import mouse
import math

# 用于保存两个像素点的坐标
point1 = None
point2 = None

def on_right_click(x, y, button, pressed):
    global point1, point2
    
    if button == mouse.Button.right:
        if not point1:
            # 第一次点击右键时记录第一个像素点的坐标
            point1 = (x, y)
            print("第一个点的坐标为:", point1)
        else:
            # 第二次点击右键时记录第二个像素点的坐标
            point2 = (x, y)
            print("第二个点的坐标为:", point2)
            
            # 计算两个点之间的距离
            distance = math.dist(point1, point2)
            print("两个点之间的距离为:", distance)

# 创建鼠标监听器
listener = mouse.Listener(on_click=on_right_click)

# 启动监听器
listener.start()

# 进入事件循环
pyautogui.PAUSE = 0.1
pyautogui.FAILSAFE = True
pyautogui.alert("请进行右键点击操作")
