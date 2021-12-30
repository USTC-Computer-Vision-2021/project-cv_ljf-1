import cv2
import numpy as np

# 处理输入图片，通过转换图片颜色空间，并通过阈值处理，高斯模糊等方式
# 除去颜色信息，提取出线条和边界信息
def image_proc(img, scale_factor):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Luminance channel
    lum = img_hsv[:,:,2]

    # 自适应阈值
    lum_thresh = cv2.adaptiveThreshold(lum,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,15)

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(lum_thresh, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    min_size = 90*scale_factor

    lum_clean = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            lum_clean[output == i + 1] = 255

    # 用一个mask除去neat outline
    lum_seg = np.copy(lum)
    lum_seg[lum_clean!=0] = 0
    lum_seg[lum_clean==0] = 255

    # 高斯平滑
    lum_seg = cv2.GaussianBlur(lum_seg,(3,3),1)

    return lum_seg

# 计算homography矩阵
def computeHomography(src_pts, dst_pts):
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    return M

# 在图片中框出模板匹配的区域
def draw_frame(img, dst):
    img = cv2.polylines(img, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)

    return img

# 使用BRISK keypoint detector寻找keypoint及descriptor
# 并用FLANN feature matching进行特征匹配
def brisk_flann(img1, img2):
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
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 50
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = computeHomography(src_pts, dst_pts)

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

    else:
        print ("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

    return dst_pts, dst

# 将立方体投影至视频帧并显示
def plot_cube(img_marked, rvecs, tvecs, camera_matrix, dist_coefs):
    # 立方体世界角点坐标
    axis8 = np.float32([[0, 0, 0], [12, 0, 0], [12, 12, 0], [0, 12, 0], [0, 0, -12], [12, 0, -12], [12, 12, -12],
                        [0, 12, -12]]).reshape(-1, 3)

    # 将立方体角点投影至视频帧
    imgpts, jac = cv2.projectPoints(axis8, rvecs, tvecs, camera_matrix, dist_coefs)

    # 调整转动立方体
    # 显示上下表面，之间的棱角用红色连接
    imgpts = np.int32(imgpts).reshape(-1, 2)
    face1 = imgpts[:4]
    face2 = np.array([imgpts[0], imgpts[1], imgpts[5], imgpts[4]])
    face3 = np.array([imgpts[2], imgpts[3], imgpts[7], imgpts[6]])
    face4 = imgpts[4:]

    # 下表面
    img = cv2.drawContours(img_marked, [face1], -1, (255, 0, 0), -3)

    # 连接上下表面
    img = cv2.line(img_marked, tuple(imgpts[0]), tuple(imgpts[4]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[1]), tuple(imgpts[5]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[2]), tuple(imgpts[6]), (0, 0, 255), 2)
    img = cv2.line(img_marked, tuple(imgpts[3]), tuple(imgpts[7]), (0, 0, 255), 2)

    # 上表面
    img = cv2.drawContours(img_marked, [face4], -1, (0, 255, 0), -3)

    return img_marked