import cv2
import numpy as np

# 棋盘格子大小
square_size = 0.02

# 棋盘格子的行数和列数
rows = 6
cols = 8

# 生成棋盘格子的三维坐标
objp = np.zeros((rows*cols, 3), np.float32)
objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

# 存储棋盘格子的三维坐标和对应的图像坐标
objpoints = []
imgpoints = []

# 读取初始帧的图像
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格子的角点
    ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # 标定相机
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # 计算相机在初始帧标定的棋盘位置后的偏移
        rvec, tvec = rvecs[0], tvecs[0]
        R, _ = cv2.Rodrigues(rvec)
        inv_R = np.linalg.inv(R)
        inv_tvec = -np.dot(inv_R, tvec)

        # 在图像上绘制偏移向量
        img = cv2.drawFrameAxes(img, mtx, dist, rvec, tvec, 0.1)

        # 显示图像
        cv2.imshow('Camera Calibration', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
