import cv2
import numpy as np
import process_func as pf

# 数据集、相机参数、模板路径
dataset_path = '../data/dataset/'
param_path = '../data/param/'
template_path = '../data/templates/'
save_path = '../output/'

template_filename = template_path + 'template.jpg'  # 模板
video_filename = dataset_path + 'colour_vid1.mp4'  # 待处理视频
camera_params_filename = param_path + 'cam_params.npz'  # 相机参数
output_filename =  save_path + 'output_3d.mp4'  # 视频保存路径

# 3D世界角点坐标
pg_points = np.array([
    (93.0, 135.0, 0.0),  # 1
    (93.0, -135.0, 0.0),  # 2
    (-93.0, -135.0, 0.0),  # 3
    (-93.0, 135.0, 0.0)  # 4
])

# Load相机内参矩阵和畸变系数
# 手机摄像头已使用张氏标定法进行标定
cam_params = np.load(camera_params_filename)
camera_matrix = cam_params['camera_matrix']
dist_coefs = cam_params['dist_coefs']

# 定义视频输出格式及输出对象
out = cv2.VideoWriter(output_filename, 0x00000021, 25.0, (960,540))

# 读取模板图片，在这里的实现中通过image_proc函数转换图片颜色空间，并通过阈值处理，高斯模糊等方式
# 除去颜色信息，提取出线条和边界信息
img_org = cv2.imread(template_filename)

# 缩小图片尺寸以减少运算量
scale_factor = 0.25
# img1 = pf.image_proc(cv2.resize(img_org,None,fx=scale_factor,fy=scale_factor),scale_factor)
img1 = pf.image_proc(cv2.resize(img_org,(540,960)),scale_factor)

# 利用cv2.VideoCapture函数获取视频帧
# 提取出第一帧并通过image_proc提取边界信息
cap = cv2.VideoCapture(video_filename)
_,img_fframe = cap.read()
img_fframe_resize = cv2.resize(img_fframe, None, fx=0.5, fy=0.5)
img2_fframe = pf.image_proc(img_fframe_resize, 0.5)


# STAGE 1 brisk_flann法提取feature points
# 特征识别比对
dst_pts, dst = pf.brisk_flann(img1, img2_fframe)

# 在视频帧中框出模板匹配的区域
img_marked = pf.draw_frame(img_fframe_resize, dst)
cv2.imshow('Video',img_marked)

# STAGE 2 光流法估计相机相机角度并在视频中插入立方体
# 保存参数，在光流法中将会用到brisk-flann法得到的feature points
src_pts = np.copy(dst_pts)
img2_old = np.copy(img2_fframe)

# Parameters for Shi-Tomasi corner detection
# feature_params = dict( maxCorners = 100,
#                        qualityLevel = 0.3,
#                        minDistance = 7,
#                        blockSize = 7 )
#
# src_pts = cv2.goodFeaturesToTrack(img2_fframe, mask = None, **feature_params)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Main Process 
# 光流法逐帧处理视频
# 求解相机位置参数并将一个立方体投影至视频帧
while True:
    # 将处理后的图片写入输出视频文件 'output.mp4'
    out.write(img_marked)

    # 读取视频帧
    ret, img_scn = cap.read()

    if ret:
        # 缩小图片尺寸以减少运算量
        img_scn_resize = cv2.resize(img_scn, None, fx=0.5, fy=0.5)

        # 读取模板图片，通过image_proc函数除去颜色信息，提取出线条和边界信息
        img2 = pf.image_proc(img_scn_resize, 0.5)

        # 计算光流
        dst_pts, st, err = cv2.calcOpticalFlowPyrLK(img2_old, img2, src_pts, None, **lk_params)

        # 找到好的对应点
        good_new = dst_pts[st == 1]
        good_old = src_pts[st == 1]

        # 计算Homography矩阵
        M = pf.computeHomography(good_old, good_new)

        # 利用Homography矩阵转换frame的边界
        dst = cv2.perspectiveTransform(dst, M)

        # 在视频帧中框出模板匹配的区域
        img_marked = pf.draw_frame(img_scn_resize, dst)

        # 保留特征点和视频帧以便在下一帧中使用光流法进行估计
        src_pts = np.copy(good_new).reshape(-1,1,2)
        img2_old = np.copy(img2)

        # 通过世界与视频帧的角点坐标估计相机位置参数
        # 旋转平移向量通过cv2.solvePnPRansac获得
        ret, rvecs, tvecs, inlier_pt = cv2.solvePnPRansac(pg_points, dst, camera_matrix, dist_coefs)

        # 调整转动立方体
        # 将立方体投影至视频帧并显示出来
        img_marked = pf.plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs)

        cv2.imshow('Video',img_marked)
      
        # 按 'q' 可退出程序
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        print ('End of video')
        break

# 关闭窗口并释放VideoCapture
cv2.destroyAllWindows()
cap.release()
