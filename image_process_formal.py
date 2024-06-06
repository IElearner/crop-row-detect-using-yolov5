#可以处理视频
from skimage.draw import line as skidline
from scipy.signal import argrelextrema, savgol_filter
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
def binary_simple_purple_rgb(img):
    # 设定紫色的阈值范围
    lower_purple_hue = 120  # 色相（Hue）值约为蓝色
    upper_purple_hue = 150  # 色相（Hue）值约为蓝色

    lower_threshold = 50  # 饱和度（Saturation）和明度（Value）的下限
    upper_threshold = 255  # 饱和度（Saturation）和明度（Value）的上限

    # 将BGR图片转换为HSV色彩空间
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # 分离色相通道
    hue_channel = hsv[:, :, 0]

    # 创建一个与原图同样大小的零矩阵，用于存储紫色区域
    purple_binary_mask = np.zeros_like(hue_channel)

    # 将色相通道中在紫色阈值范围内、饱和度和明度在阈值范围内的像素设为255（白色）
    purple_binary_mask[
        (hue_channel >= lower_purple_hue) & (hue_channel <= upper_purple_hue) &
        (hsv[:, :, 1] >= lower_threshold) & (hsv[:, :, 2] >= lower_threshold)
    ] = 255

    # 将其余像素设为0（黑色）
    purple_binary_mask[
        ~((hue_channel >= lower_purple_hue) & (hue_channel <= upper_purple_hue) &
          (hsv[:, :, 1] >= lower_threshold) & (hsv[:, :, 2] >= lower_threshold))
    ] = 0

    return purple_binary_mask

def find_best_counts(binary_img):  #主要费时函数
    # Get shape of image
    (height, width) = binary_img.shape[:2]
    # Initialize arrays to hold counts and top points
    best_count = np.zeros(width)
    best_top = np.zeros(width)
    # Loop through bottom and top points
    for bottom in range(width):
        for top in range(width):
            # Calculate line between 2 given pixels
            rr, cc = skidline(height - 1, bottom, 0, top)

            # Count nonzero (green) points along line
            # divide by total number of points along line
            count = cv.countNonZero(binary_img[rr, cc]) / len(rr)

            # Set new best line for this bottom point
            if count > best_count[bottom]:
                best_top[bottom] = top
                best_count[bottom] = count

    return best_top, best_count

def draw_peak_lines_red_plus(img, count, top, height):

    peaks = argrelextrema(
        savgol_filter(count, 20, 5), np.greater, order=int(len(count) / 5)
    )
    #print(peaks)
    min_length = float('inf')
    shortest_peak = None


    # Find the shortest peak
    for bottom_peak in peaks[0]:
        length = abs(height - 1 - 0) + abs(bottom_peak - top[int(bottom_peak)])
        if length < min_length:
            min_length = length
            shortest_peak = bottom_peak

    if shortest_peak != None:
        shortest_start = (int(shortest_peak), height - 1)
        shortest_end = (int(top[int(shortest_peak)]), 0)
        # print("Shortest line coordinates:", shortest_start, shortest_end)

        # For each row estimate, draw a line on the image
        for bottom_peak in peaks[0]:
            color = (0, 0, 255) if bottom_peak == shortest_peak else (255, 0, 0)
            cv.line(
                img,
                (int(bottom_peak), height - 1),
                (int(top[int(bottom_peak)]), 0),
                color,
                2,
            )

    else:
        shortest_start = None
        shortest_end = None

    return shortest_start, shortest_end

def get_matrix(a,b):
    original_points = np.float32(a)
    transformed_points = np.float32(b)
    perspective_matrix = cv.getPerspectiveTransform(original_points, transformed_points)
    return perspective_matrix

def transform_perspective(image,start,end,perspective_matrix): #(x1,y1)(x2,y2)为原始图中间线也就是目标划线的起始点

    # 读取图片
    (img_height, img_width) = image.shape[:2]
    # 进行透视变换
    #transformed_image = cv.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))
    (tr_height, tr_width) = (img_height, img_width)
    #print(f"当前位置坐标为：({tr_width / 2},{tr_height}),方向角度为90度")

    if start != None:
        x1 = start[0]
        y1 = start[1]
        x2 = end[0]
        y2 = end[1]
        # 输入点
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        # 添加齐次坐标
        points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        # 应用透视变换
        transformed_points = np.dot(perspective_matrix, points.T).T

        # 归一化齐次坐标
        transformed_points[:, 0] /= transformed_points[:, 2]
        transformed_points[:, 1] /= transformed_points[:, 2]

        # 去除齐次坐标
        transformed_points = transformed_points[:, :2]
        # print(transformed_points)
        point2 = transformed_points[0]
        point1 = transformed_points[1]
        # print(f"直接变换后的点坐标:({point1[0]},{point1[1]}),"
        # f"({point2[0]},{point2[1]})")

        # 计算线段的斜率
        if point2[0] - point1[0] != 0:
            m = (point2[1] - point1[1]) / (point2[0] - point1[0])
        else:
            m = float('inf')  # 无限斜率，线段垂直于 x 轴

        # 计算线段方程式中的截距
        if m != float('inf'):  # 如果不是垂直线
            c = point1[1] - m * point1[0]
        else:
            c = None  # 垂直线的截距为 None

        # 交点列表
        intersections = []

        # 计算与图像上边界的交点
        x = (0 - c) / m if m != 0 else 0
        if 0 <= x <= img_width:
            intersections.append((x, 0))

        # 计算与图像下边界的交点
        x = (img_height - 1 - c) / m if m != 0 else point1[0]
        if 0 <= x <= img_width:
            intersections.append((x, img_height - 1))

        # 计算与图像左边界的交点
        if m != 0:
            y = m * 0 + c
            if 0 <= y <= img_height:
                intersections.append((0, y))

        # 计算与图像右边界的交点
        if m != 0:
            y = m * (img_width - 1) + c
            if 0 <= y <= img_height:
                intersections.append((img_width - 1, y))

        # print(f"转换后起始点为：{intersections}")

        tan_angle = - (intersections[1][1] - intersections[0][1]) / (intersections[1][0] - intersections[0][0])

        angle_x = math.atan(tan_angle)

        # 将角度转换为度数
        angle_x_degrees = math.degrees(angle_x)
        if angle_x_degrees < 0:
            angle_x_degrees = angle_x_degrees + 180
        print(f"目标位置相对坐标为：(k*{intersections[1][0] - tr_width / 2},0)，k为比例尺，目标角度为{angle_x_degrees}度")
    else:
        print("未找到作物行")

def paint(cb):
    # 绘制滤波前图像(底部）
    sgcb = savgol_filter(cb, 15, 5)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.plot(cb, label='原始数据')
    plt.plot(sgcb, label='SG平滑后数据')
    plt.legend()
    plt.show()


def your_image_processing_function(img,perspective_matrix):
    # 转换为 HSV 并计算二值图像
    binary_img = binary_simple_purple_rgb(img)
    print('1')
    # hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # 计算图像的高度和宽度
    (height, width) = binary_img.shape[:2]
    # 找到每个底部像素的最佳行估计
    top, count = find_best_counts(binary_img)
    print('2')
    # 绘制滤波图
    # paint(count)
    # 找到行的最佳估计并添加到图像中
    start, end = draw_peak_lines_red_plus(img, count, top, height)
    print('3')
    # 透视变换，给出角度
    #transform_perspective(img, start, end,perspective_matrix)
    return img


