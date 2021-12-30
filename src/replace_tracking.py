import cv2
import numpy as np
import process_func as pf

# 数据集、相机参数、模板路径
dataset_path = '../data/dataset/'
param_path = '../data/param/'
template_path = '../data/templates/'
save_path = '../output/'

template_filename = template_path + 'template.jpg'  # 模板
video_filename = dataset_path + 'colour_vid.mp4'  # 待处理视频
camera_params_filename = param_path + 'cam_params.npz'  # 相机参数
output_filename = save_path + 'output_replace.mp4'  # 视频保存路径
target_filename = template_path + 'template1.jpg'


# Load相机内参矩阵和畸变系数
# 手机摄像头已使用张氏标定法进行标定
cam_params = np.load(camera_params_filename)
camera_matrix = cam_params['camera_matrix']
dist_coefs = cam_params['dist_coefs']

# 定义视频输出格式及输出对象
out = cv2.VideoWriter(output_filename, 0x00000021, 25.0, (960,540))

# 读取模板图片，在这里的实现中通过image_proc函数转换图片颜色空间，并通过阈值处理，高斯模糊等方式
# 除去颜色信息，提取出线条和边界信息
template = cv2.imread(template_filename)
# 缩小图片尺寸以减少运算量
scale_factor = 0.25
# img1 = pf.image_proc(cv2.resize(template,None,fx=scale_factor,fy=scale_factor),scale_factor)
img1 = pf.image_proc(cv2.resize(template,(540,960)),scale_factor)

img_target = cv2.imread(target_filename)
img_target = cv2.resize(img_target, (540,960))

# 利用cv2.VideoCapture函数获取视频帧
cap = cv2.VideoCapture(video_filename)
frame_count = 0

# Main Process 
# 逐帧处理视频并用目标图片覆盖模板
while True:

    # 读取视频帧
    ret, img_scn = cap.read()
    frame_count = frame_count + 1
    if ret:

        img_resize = cv2.resize(img_scn, None, fx=0.5, fy=0.5)
        img2 = pf.image_proc(img_resize, 0.5)

        # dst_pts, dst = pf.brisk_flann(img1, img2)
        # BRISK keypoint detector
        brisk = cv2.BRISK_create()
        # 用BRISK detector找到模板图片和及视频帧的keypoint及descriptor
        kp1, des1 = brisk.detectAndCompute(img1, None)
        kp2, des2 = brisk.detectAndCompute(img2, None)

        # FLANN特征匹配参数
        FLANN_INDEX_LSH = 1
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=1)  

        # FLANN特征匹配
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 判断是否为好的匹配
        good = []
        try:
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)
        except ValueError:
            pass

        MIN_MATCH_COUNT = 40
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
        
            # 利用Homography矩阵进行投影
            imgWarp = cv2.warpPerspective(img_target,M,(img_resize.shape[1],img_resize.shape[0]))
            # 降噪，去掉最大或最小的像素点
            retval, threshold_img = cv2.threshold(imgWarp, 0, 255, cv2.THRESH_BINARY)
            imgWarp = imgWarp + threshold_img

            whiteMask = np.zeros((img_resize.shape[0],img_resize.shape[1]),np.uint8)
            cv2.fillPoly(whiteMask,[np.int32(dst)],(255,255,255))   
            blackMask = cv2.bitwise_not(whiteMask)

            img_mask = cv2.bitwise_and(img_resize,img_resize,mask=blackMask)
            img_resize = cv2.bitwise_or(img_mask,imgWarp)

        # Display in video
        cv2.imshow('Video', img_resize)
      
        # 按 'q' 可退出程序
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        # 将处理后的图片写入输出视频文件 'output.mp4'
        out.write(img_resize)

    else:
        print ('End of video')
        break

# 关闭窗口并释放VideoCapture
cv2.destroyAllWindows()
cap.release()
