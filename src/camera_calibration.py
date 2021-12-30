import os
import numpy as np
import cv2
import glob

# 相机参数标定得到相机的内参矩阵，畸变矩阵
def calib(inter_corner_shape, size_per_grid, img_dir, img_type, save_coner_dir):
    # 标准：仅用于subpix校准，此处不使用。
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    w,h = inter_corner_shape
    # cp_int：int形式的角点，以int形式保存世界空间中角点的坐标
    # like (0,0,0), (1,0,0), (2,0,0) ....,(w,h,0).
    cp_int = np.zeros((w*h,3), np.float32)
    cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2) # 将世界坐标系建在标定板上，所有点的z坐标全部为0
    # cp_world：世界空间中的角点，保存世界空间中角点的坐标。
    cp_world = cp_int*size_per_grid
    
    obj_points = [] # 3D世界空间中的点
    img_points = [] # 2D图像空间中的点（与obj_points相关）
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # 找到角点corners：像素空间中的角点。
        ret, corners = cv2.findChessboardCorners(gray_img, (w,h), None)
        # if ret is True, save.
        if ret == True:
            obj_points.append(cp_world)
            # 用于subpix校准，此处可不使用
            # corners2 = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1,-1), criteria)
            # if [corners2]:
            #     img_points.append(corners2)
            # else:
            #     img_points.append(corners)
            img_points.append(corners)
            # 显示角点
            cv2.drawChessboardCorners(img, (w,h), corners, ret)
            # cv2.imshow('FoundCorners',img)
            # cv2.waitKey(1)
            cv2.imwrite(save_coner_dir + os.sep + img_name, img)   
    print('Corner images have been saved to: %s successfully.' %save_coner_dir)         
            
    cv2.destroyAllWindows()
    # 校准相机
    ret, inter_matrix, dist_coefs, v_rot, v_trans = cv2.calibrateCamera(obj_points, img_points, gray_img.shape[::-1], None, None)
    np.savez('../data/param/cam_params', ret=ret, camera_matrix=inter_matrix, dist_coefs=dist_coefs)
    # print (("ret:"),ret)
    # print (("internal matrix:\n"),inter_matrix)
    # # in the form of (k_1,k_2,p_1,p_2,k_3)
    # print (("distortion cofficients:\n"),dist_coefs)  
    # print (("rotation vectors:\n"),v_rot)
    # print (("translation vectors:\n"),v_trans)
    # 计算重新投影的误差
    total_error = 0
    for i in range(len(obj_points)):
        img_points_repro, _ = cv2.projectPoints(obj_points[i], v_rot[i], v_trans[i], inter_matrix, dist_coefs)
        error = cv2.norm(img_points[i], img_points_repro, cv2.NORM_L2)/len(img_points_repro)
        total_error += error
    print(("Average Error of Reproject: "), total_error/len(obj_points))
    
    return inter_matrix, dist_coefs

# 根据得到的内参矩阵，畸变矩阵来进行图像纠偏	
def dedistortion(inter_corner_shape, img_dir,img_type, save_dir, inter_matrix, dist_coefs):
##   w,h = inter_corner_shape
      
##   img2 = cv2.imread(img_dir_sin)
    (w,h) = (640,480)
##   newcameramtx, roi = cv2.getOptimalNewCameraMatrix(inter_matrix,dist_coefs,(w,h),0,(w,h))
##   dst = cv2.undistort(img2, inter_matrix, dist_coefs, None, newcameramtx)
##   cv2.imshow('dst',dst)
    
    images = glob.glob(img_dir + os.sep + '**.' + img_type)
    for fname in images:
        img_name = fname.split(os.sep)[-1]
        img = cv2.imread(fname)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(inter_matrix,dist_coefs,(w,h),0,(w,h)) # 自由比例参数
        dst = cv2.undistort(img, inter_matrix, dist_coefs, None, newcameramtx)
        # clip the image
        # x,y,w,h = roi
        # dst = dst[y:y+h, x:x+w]
        # cv2.imshow('dst',dst)
        # cv2.waitKey()
        cv2.imwrite(save_dir + os.sep + img_name, dst)

    print('Dedistorted images have been saved to: %s successfully.' %save_dir)
    
if __name__ == '__main__':
	# 棋盘格格点(9-1,6-1)
    inter_corner_shape = (8,5)
	# 格点尺寸26mm
    size_per_grid = 0.026
	# 标定照片文件夹
    img_dir = "../data/param/camera_calib_img"
    img_type = "jpg"
	# 保存识别角点文件夹
    save_coner_dir = "../data/param/save_coner"
    # 相机标定
    inter_matrix, dist_coefs = calib(inter_corner_shape, size_per_grid, img_dir,img_type, save_coner_dir)
    # 保存矫正后图像文件夹 
    save_dir = "../data/param/save_dedistortion"
    
    if(not os.path.exists(save_dir)):
        os.makedirs(save_dir)
    dedistortion(inter_corner_shape, img_dir, img_type, save_dir, inter_matrix, dist_coefs)