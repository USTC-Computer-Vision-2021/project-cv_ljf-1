# 基于OpenCV库的对象替代及3D实时成像

成员分工

- 骆霁飞 PB18000195
  - 设计，编程，报告

## 问题描述

- Augmented Reality (AR) 增强现实技术可以将真实世界信息和虚拟世界信息内容相互叠加，从而实现超越现实的视觉效果。在一些电视魔术表演中，魔术师似乎可以凭空变出纸牌或者让纸牌消失，在近景魔术中他们更多会依靠手法，但在电视魔术中他们通常会选择“黑桌布”来实现更震撼的效果，这个所谓“黑桌布”就是通过一系列机关实现藏牌的动作，但因为这种魔术常常是摄像头俯视拍摄，并得益于桌布良好的吸光特性，电视机前的观众根本看不出桌面的任何变化，既然都是欺骗观众眼睛，受AR技术的启发，我基于OpenCV库实现了一个对象遮盖替代的AR算法`src/replace_tracking.py`，通过适时调用该算法，我们也可以实现一个简单的“换牌魔术”。
- 通常AR成像需要进行关键点的提取和匹配，逐帧进行对目标图像与视频帧的关键点匹配效率比较低，在算力不足的情况下难以实现实时的匹配，为改善这个问题我基于OpenCV库使用Lucas Kanade光流算法，通过使用上一帧的关键点信息来对下一帧进行估计，从而大幅提高计算速度并实现实时成像。基于同样目标的视频素材，我在`src/3d_tracking.py`中实现了一个3D立方体的实时成像算法，这里我没有使用现成的obj模型，而绘制了一个立方体框架以便展示空间透视效果。

## 算法实现

我的程序实现中需要的主要功能及使用的OpenCV库函数包括：

1. 相机标定 Camera calibration (`cv2.findChessboardCorners`, `cv2.calibrateCamera`)
2. 图片预处理：线条信息提取，平滑及阈值处理 Extraction of line drawing, smoothing and thresholding (`cv2.GaussianBlur`,`cv2.adaptiveThreshold`, `cv2.connectedComponentsWithStats`)
3. BRISK特征提取器 BRISK feature detector (`cv2.BRISK_create`, `brisk.detectAndCompute`)
4. FLANN特征匹配 FLANN feature matching (`cv2.FlannBasedMatcher`, `flann.knnMatch`)
5. Lucas Kanade光流跟踪 Optical flow tracking (`cv2.calcOpticalFlowPyrLK`)
6. 计算单应矩阵 Homography (`cv2.findHomography`)
7. 相机姿态估计 Camera pose estimation (`cv2.solvePnPRansac`)
8. 立方体绘制 Cube plotting  (`cv2.line`, `cv2.drawContours`)

在`augmented_reality.ipynb`有主要算法步骤的代码示例

### 图片预处理

不论是实现3D跟踪还是目标替换，很关键的一步就是要匹配模板与视频帧中的目标，通常情况下可以在RGB空间中直接使用ORB算法或BRISK算法进行特征匹配，但在课上我受到老师启发：因为我们寻找的特征点通常出现在边界和角落，颜色信息在特征寻找匹配过程中并不重要，因此我们可以先对原始图像做一个颜色空间转化到HSV空间，在这个空间里我们就可以主要关注亮度信息了，OpenCV库提供了`cv2.adaptiveThreshold`函数求取自适应阈值进行二值化，以及`cv2.connectedComponentsWithStats`函数求取连通域，我们可以较为方便地调取这些函数从而提取出关键的线条信息，在`data/process_func.py`中的`image_proc`函数中我实现了这一操作，下面左图是模板图片经`image_proc`函数处理之后的效果图，右图是特征提取得到的keypoints。

<img src="output\line_extract.png" alt="line_extract" style="zoom:80%;" />

### 特征提取匹配

为了追踪视频中的目标，必不可少的一步便是特征提取匹配，OpenCV库里提供了ORB算法供直接调用，我使用模板图片在RGB空间内对ORB算法的keypoints检测效果进行了测试，得到的结果如下：

<img src="output\orb_keypoint.png" alt="orb_keypoint" style="zoom:80%;" />

可以看出ORB算法的确找到了很多特征点，但似乎仍旧不够令人满意，为此我经过调查，了解到OpenCV库里提供了表现更好的BRISK特征提取器，特征点匹配算法也有比BF法更灵活的`cv2.FlannBasedMatcher`，下面是BRISK feature detector对keypoints的检测效果：

<img src="output\brisk_keypoint.png" alt="brisk_keypoint" style="zoom:80%;" />

我们可以通过计算比较匹配点的distance来判断匹配的优劣程度，可以通过设置阈值只保留匹配度最高的一些角点来为跟踪，遮盖，3D建模做准备，下面是选取匹配度最高的一些角点利用`cv2.drawMatches`作出的匹配图：

<img src="output\keypoint_match.png" alt="keypoint_match" style="zoom:80%;" />

### 模板遮盖

在上面的步骤中，我们已经用ORB或BRISK特征提取器找到了关键点并进行了匹配，接下来要做的就是将我们的目标图片覆盖到视频帧中，我们需要一种算法可以帮助我们将目标图片与视频帧中的对应位置做一个映射，幸运的是，单应矩阵Homography给出了解决方法，简单来说，从一个平面到另外一个平面做一对一投影映射被称做单应，单应矩阵是一个 $3\times3$ 的矩阵，一个平面上点的坐标乘上单应矩阵即可映射到目标平面的指定位置，OpenCV库提供的`cv2.findHomography`函数可以通过模板与视频帧的关键点估计Homography矩阵，得到这个矩阵之后我们便可模板遮盖，具体代码可参见`augmented_reality.ipynb`和`replace_tracking.py`，下面是处理得到的图片，将这一算法应用到整个视频即可实现最终的结果。

<img src="output\replace.png" alt="replace" style="zoom:80%;" />

### 立方体绘制

这一步是在视频中显示一个3D立方体，因为投影的是一个立体图形，在Homography矩阵的基础上，我们结合相机内参矩阵和畸变系数等信息计算出三维空间中的投影矩阵，相机的内参系数通过张氏标定法标定，算法程序在`camera_calibration.py`中，利用该投影矩阵，我们在世界坐标系下定义一个立方体的角点坐标，然后可以利用OpenCV中的`cv2.projectPoints`函数将立方体角点投影至视频帧，再使用`cv2.drawContours`以及`cv2.line`等函数在视频帧中绘制出立方体轮廓，该立方体就会随着摄像头视角转换进行相应的旋转，相机姿态的估计可以用OpenCV中的`cv2.solvePnPRansac`实现。在逐帧进行对目标图像与视频帧的关键点匹配时，我发现该方式效率太低，在算力不足的情况下难以实现实时的匹配，为改善这个问题我基于OpenCV的Lucas Kanade光流算法来对临近帧关键点进行估计，这样可以充分利用上一帧的特征点信息，大幅提高计算速度从而实现实时成像，在`3d_tracking.py`中我使用Lucas Kanade光流算法实现了小立方体的实时绘制，下面是立方体显示效果图：

<img src="output\3d_cube.png" alt="3d_cube" style="zoom:80%;" />

## 效果展示

这幅动图是应用`src/replace_tracking.py`算法，使用目标卡牌替代原视频中卡牌得到的结果

<img src="output\replace_tracking.gif" alt="replace_tracking" style="zoom:80%;" />

这幅动图是应用`src/3d_tracking.py`3D立方体实时成像算法得到的结果

<img src="output\3d_tracking.gif" alt="3d_tracking" style="zoom:80%;" />

## 工程结构

```
.
├── README.md
├── requirements.txt
├── augmented_reality.ipynb
├── src
│   ├── 3d_tracking.py
│   ├── replace_tracking.py
│   ├── camera_calibration.py
│   └── process_func.py
├── data
│   ├── dataset
│   │	├── colour_vid.mp4
│   │	└── colour_vid1.mp4
│   ├── param
│   │	├── camera_calib_img
│   │	├── save_coner
│   │	├── save_dedistortion
│   │	└── cam_params.npz
│   └── templates
│   	└── template.jpg
└── output
	├── output_3d.mp4
    └── output_replace.mp4
```

## 运行说明

这是我的实验运行环境，复现时只要版本号不太低应该都能正常运行

```
numpy==1.14.3 
opencv-python==4.4.0.46
matplotlib==2.2.2
```

运行时可在命令行执行以下操作，起始位置在主文件夹

```
cd src
python 3d_tracking.py
python replace_tracking.py
```

如果需要使用自己的相机进行标定，需先在`data/param/camera_calib_img`放上10至20张棋盘纸照片，并在`camera_calibration.py`中设置角点个数及棋格大小，在主文件夹中执行以下命令

```
cd src
python camera_calibration.py
```

即可在`data/param/cam_params.npz`中保存下相机的内参矩阵，畸变矩阵，同时在`data/param/save_coner`和`data/param/save_dedistortion`文件夹里则分别保存了识别出了角点的照片以及矫正后的照片。
