👨‍💻 Facial Feature Extractor - 自动化面部特征提取与分析系统
<p align="center">
<a href="https://github.com/laicai0810/facial_feature/tree/main"><img src="https://img.shields.io/badge/GitHub-Repo-blue.svg" alt="GitHub Repo"></a>
<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python版本">
<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="许可证">
<img src="https://img.shields.io/badge/Dlib-19.24-orange.svg" alt="Dlib版本">
<img src="https://img.shields.io/badge/OpenCV-4.x-blue.svg" alt="OpenCV版本">
</p>

一套功能全面的Python库，用于从图像中检测人脸、提取面部关键点、执行图像增强，并计算超过150种丰富的几何面部特征。本项目基于强大的 Dlib 和 OpenCV 库构建，旨在为下游的机器学习任务（如情绪识别、生物特征分析、金融风控等）提供高维度、结构化的特征数据。

✨ 核心功能
👨 人脸检测: 采用Dlib的HOG特征+线性SVM分类器，高效定位正面人脸区域。

📍 关键点预测: 基于回归树集成算法，精准提取68个面部关键点。

📐 面部对齐: 利用关键点信息生成姿态归一化的标准尺寸面部图像，消除头部姿态变化带来的影响。

🎨 图像增强: 提供一套可选的图像预处理流程，包括光照归一化 (CLAHE)、灰度转换和图像降噪 (双边滤波)等。

📊 特征计算: 自动计算超过150种面部几何特征，涵盖距离、角度、比率、对称性与轮廓面积等多个维度。

🗺️ 核心处理流程
下图直观地展示了单张图片输入系统后，从原始像素到最终结构化特征数据的完整处理流程。

graph TD
    A[🖼️ 输入原始图像] --> B{1. 人脸检测};
    B --> |检测到人脸| C(dlib.rectangle);
    B --> |未检测到| F[❌ 输出: 无人脸];
    C --> D{2. 关键点定位};
    D --> |定位成功| E[📍 68个面部关键点];
    D --> |定位失败| G[❌ 输出: 关键点错误];
    E --> H{3. 面部对齐};
    H --> I[👤 标准化面部切片 (Chip)];
    I --> J{4. 图像增强 (可选)};
    J --> K[✨ 增强后的图像];
    E & K --> L{5. 几何特征计算};
    L --> M[📊 150+ 结构化特征];
    M --> N[💾 输出: 特征数据 (CSV)];

    subgraph "核心处理模块 (facial_feature_extractor)"
        B; C; D; E; H; I; J; K; L; M;
    end

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style N fill:#ccf,stroke:#333,stroke-width:2px

🔬 关键技术深度解析
步骤

核心模块

技术细节

可视化参考

1. 人脸检测

detection.py

技术: Dlib内置的基于方向梯度直方图 (HOG) 特征和线性支持向量机 (SVM) 的分类器。<br>流程: 灰度转换 -> 构建图像金字塔 -> 滑动窗口计算HOG特征 -> SVM分类 -> 选择面积最大的人脸。



2. 关键点定位

landmarks.py

技术: Dlib的 shape_predictor，一种基于回归树集成的快速算法。<br>模型: 使用预训练的shape_predictor_68_face_landmarks.dat模型。<br>输出: 68个(x, y)坐标点，精确勾勒出面部轮廓。



3. 面部对齐

landmarks.py

技术: 调用dlib.get_face_chip函数。<br>流程: 内部执行相似性变换（旋转、缩放、平移），将眼睛和鼻子置于标准位置，生成一张固定大小、姿态基本校正后的人脸图像，为后续特征计算提供规范化输入。



4. 图像增强

enhancement.py

光照归一化: 采用限制对比度的自适应直方图均衡化 (CLAHE) 算法，改善光照不均问题。<br>降噪处理: 使用双边滤波器 (Bilateral Filter)，在平滑图像的同时能够很好地保留边缘信息。



5. 特征计算

features.py

基础: 所有特征的计算都基于第2步得到的68个关键点的坐标。<br>特征类别: 距离（瞳孔间距）、比率（眼宽高比EAR）、角度（下颌角）、面积/周长（嘴唇轮廓）等。



📈 特征维度详解
本系统共提取超过150个几何特征，以下为关键特征维度的分类与详细说明：

1. 面部轮廓与比例 (Facial Contour & Proportions)
这些特征描述了脸部的基本形状、大小和整体比例。

face_width_max_jaw: 下颌最宽处距离，衡量脸部宽度。

face_height_nose_bridge_to_chin: 脸长，从鼻梁顶部到下巴尖的垂直距离。

face_width_to_height_ratio: 脸的宽高比，是判断脸型（如圆脸、长脸）的重要指标。

jawline_length: 下颌线轮廓的总长度。

jaw_polygon_area: 由下颌线和两端点连线构成的多边形面积。

chin_angle: 由下巴两侧点与下巴尖构成的角度，反映下巴的尖锐程度。

forehead_proxy_height: 额头高度的代理值（眉心到鼻梁顶部）。

middle_third_height_glabella_to_subnasale: 中庭高度（眉心到鼻底）。

lower_third_height_subnasale_to_chin: 下庭高度（鼻底到下巴）。

face_thirds_ratio_*: 上庭、中庭、下庭之间的高度比例，用于“三庭”分析。

2. 眼部特征 (Eye Features)
精细刻画眼睛的大小、形状和位置。

eye_width_right / eye_width_left: 左右眼的宽度（内外眼角距离）。

eye_vertical_height_right / eye_vertical_height_left: 左右眼的垂直高度。

eye_aspect_ratio_right / eye_aspect_ratio_left (EAR): 经典的眼宽高比，对眨眼检测和疲劳度分析非常敏感。

avg_ear: 左右眼EAR的平均值。

inter_ocular_distance_inner: 内眼角间距。

inter_ocular_distance_outer: 外眼角间距。

eye_area_right / eye_area_left: 左右眼轮廓所围成的面积。

five_eyes_metric_1: 内眼角间距与平均眼宽的比值，用于“五眼”比例分析。

3. 眉毛特征 (Eyebrow Features)
描述眉毛的长度、弯曲度、位置和倾斜状态。

eyebrow_length_right / eyebrow_length_left: 左右眉毛的长度。

eyebrow_arch_height_right / eyebrow_arch_height_left: 眉峰相对于眉毛两端连线的高度，反映眉毛的弯曲程度。

eyebrow_tilt_angle_right / eyebrow_tilt_angle_left: 眉毛的倾斜角度。

eyebrow_to_eye_dist_right / eyebrow_to_eye_dist_left: 眉毛到眼睛的平均距离。

4. 鼻部特征 (Nose Features)
量化鼻子的大小和形状。

nose_length: 鼻长（鼻梁顶部到鼻尖）。

nose_width_nostrils: 鼻翼宽度。

nose_length_to_width_ratio: 鼻子的长宽比。

nose_bridge_tilt_angle: 鼻梁的倾斜角度。

5. 嘴部特征 (Mouth Features)
描述嘴唇的尺寸、形状和人中区域。

mouth_width_corners: 嘴角宽度。

mouth_height_outer_lips_center: 外唇在中心点的垂直高度。

mouth_aspect_ratio_outer: 外唇的宽高比。

upper_lip_thickness_center: 上唇厚度。

lower_lip_thickness_center: 下唇厚度。

philtrum_length: 人中长度。

6. 情绪与微表情几何线索 (Emotion & Micro-expression Cues)
这些特征旨在捕捉与情绪相关的细微面部肌肉变化，对于欺诈识别等场景尤为关键。

tension_eyebrow_gap_horizontal_dist: 眉间水平距离，紧张或皱眉时通常会缩小。

tension_lip_press_ratio: 嘴唇紧绷度（垂直距离/水平距离），抿嘴时该比值会变小。

tension_jaw_clench_metric: 下巴收紧度代理值，通过下颌角到下巴尖的距离衡量。

brow_lower_intensity_y_diff: 皱眉强度，通过内外眉毛的垂直高度差计算。

anger_lip_corner_pull_down_avg_y: 嘴角下拉程度，与愤怒或悲伤情绪相关。

smile_lip_corner_pull_up_avg_y: 嘴角上扬程度，微笑的关键指标。

smile_cheek_raise_proxy_*: 脸颊肌肉隆起代理值，衡量“苹果肌”的上抬，是真实微笑（杜胥内微笑）的重要组成部分。

🚀 安装指南
克隆仓库:

git clone https://github.com/laicai0810/facial_feature.git
cd facial_feature

创建并激活Python虚拟环境:

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

安装依赖:
强烈建议先单独安装dlib，因为它可能需要编译环境。

# (可能需要先安装 cmake: pip install cmake)
pip install dlib
pip install -r requirements.txt

下载模型文件:

从官网下载 shape_predictor_68_face_landmarks.dat.bz2。

解压后，将 .dat 文件放置于项目根目录下的 models/ 文件夹内。

⚙️ 使用样例
处理单张图片
以下代码展示了如何调用核心FaceAnalyzer类来处理一张图片，并获取所有分析结果。

from facial_feature_extractor.analysis import FaceAnalyzer
from facial_feature_extractor.utils import save_image, draw_landmarks_on_image
import pprint

# 1. 初始化分析器，指定dlib模型路径
analyzer = FaceAnalyzer(shape_predictor_path='models/shape_predictor_68_face_landmarks.dat')

# 2. 调用核心方法处理图片
image_path = 'path/to/your/image.jpg' # <-- 替换为你的图片路径
result = analyzer.process_image(image_path)

# 3. 查看和使用结果
print(f"处理状态: {result['status']}")
if result['status'] == 'success':
    print(f"检测到的人脸面积: {result['face_area']}")

    # 打印部分计算出的特征
    print("\n--- 部分特征展示 ---")
    features_to_show = {k: v for k, v in result['features'].items() if 'eye_aspect' in k or 'mouth' in k}
    pprint.pprint(features_to_show)

    # 在最终处理后的图像上绘制关键点并保存
    if result['final_image'] is not None:
        final_img_with_landmarks = draw_landmarks_on_image(result['final_image'], result['landmarks'])
        save_image(final_img_with_landmarks, 'output_with_landmarks.png')
        print("\n已保存带关键点的处理后图像至 'output_with_landmarks.png'")

批量处理图片
项目scripts/目录下提供了2_run_batch_processing.py脚本，用于处理整个文件夹的图片并将所有结果汇总到一个CSV文件中。

将待处理的图片放入指定文件夹（如 data/input_images/）。

根据需要修改脚本顶部的配置变量。

执行脚本:

python scripts/2_run_batch_processing.py

📁 项目结构
.
├── facial_feature_extractor/   # 核心库代码
│   ├── __init__.py
│   ├── analysis.py             # 主分析器类
│   ├── detection.py            # 人脸检测模块
│   ├── enhancement.py          # 图像增强模块
│   ├── features.py             # 特征计算模块
│   ├── landmarks.py            # 关键点定位与对齐模块
│   └── utils.py                # 工具函数
├── scripts/                    # 示例脚本
│   ├── 1_download_images.py
│   └── 2_run_batch_processing.py
├── models/                     # 存放dlib模型文件
│   └── shape_predictor_68_face_landmarks.dat
├── .gitignore
├── README.md
└── requirements.txt

📜 许可证
本项目采用 MIT License 开源许可证。
