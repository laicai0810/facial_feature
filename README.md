# Facial Feature Extractor - 自动化面部特征提取与分析系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Dlib-19.24-orange.svg" alt="Dlib Version">
  <img src="https://img.shields.io/badge/OpenCV-4.x-blue.svg" alt="OpenCV Version">
</p>

一套功能全面的Python库，用于从图像中检测人脸、提取面部关键点、执行图像增强，并计算超过150种丰富的几何面部特征。本项目基于强大的 **Dlib** 和 **OpenCV** 库构建，旨在为下游的机器学习任务（如情绪识别、生物特征分析等）提供高维度、结构化的特征数据。

---

## ✨ 核心功能

* **👨 人脸检测**: 采用Dlib的HOG特征+线性SVM分类器，高效定位正面人脸区域。
* **📍 关键点预测**: 基于回归树集成算法，精准提取68个面部关键点。
* **📐 面部对齐**: 利用关键点信息生成姿态归一化的标准尺寸面部图像，消除头部姿态变化带来的影响。
* **🎨 图像增强**: 提供一套可选的图像预处理流程，包括光照归一化 (CLAHE)、灰度转换和图像降噪 (双边滤波)等。
* **📊 特征计算**: 自动计算超过150种面部几何特征，涵盖距离、角度、比率、对称性与轮廓面积等多个维度。

---

## 🗺️ 核心处理流程

下图直观地展示了单张图片输入系统后，从原始像素到最终结构化特征数据的完整处理流程。

```mermaid
graph TD
    A[🖼️ 输入原始图像] --> B{1. 人脸检测};
    B --> |检测到人脸| C(dlib.rectangle);
    B --> |未检测到| F[输出: 无人脸];
    C --> D{2. 关键点定位};
    D --> |定位成功| E[📍 68个面部关键点];
    D --> |定位失败| G[输出: 关键点错误];
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
🛠️ 技术实现细节
人脸检测 (detection.py):

技术: 使用Dlib内置的基于方向梯度直方图 (HOG) 特征和线性支持向量机 (SVM) 的分类器。

流程: 将图像转为灰度图 -> 构建图像金字塔以实现多尺度检测 -> 在滑动窗口上计算HOG特征 -> SVM分类 -> 选择面积最大的人脸。

关键点定位 (landmarks.py):

技术: Dlib的 shape_predictor，它是一种基于回归树集成的快速算法。

模型: 使用预训练的shape_predictor_68_face_landmarks.dat模型。

输出: 68个(x, y)坐标点，精确勾勒出面部轮廓。

面部对齐与图像增强 (landmarks.py, enhancement.py):

面部对齐: 调用dlib.get_face_chip函数，执行相似性变换（旋转、缩放、平移），生成姿态校正后的人脸图像。

光照归一化: 采用限制对比度的自适应直方图均衡化 (CLAHE) 算法，改善光照不均问题。

降噪处理: 使用双边滤波器 (Bilateral Filter)，在平滑图像的同时能够很好地保留边缘信息。

特征计算 (features.py):

基础: 所有特征的计算都基于第二步得到的68个关键点的坐标。

特征类别:

距离特征: 计算两点间的欧氏距离（如：瞳孔间距）。

比率特征: 将不同距离特征相除，以消除缩放影响（如：眼宽高比 EAR）。

角度特征: 利用余弦定理计算由三点构成的夹角（如：下颌角）。

面积/周长特征: 将多个点连接成多边形，计算其面积或周长。

🚀 安装指南
克隆仓库:

Bash

git clone <your-repo-url>
cd <your-repo-directory>
创建并激活Python虚拟环境:

Bash

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
安装依赖:
强烈建议先单独安装dlib，因为它可能需要编译环境。

Bash

# (可能需要先安装 cmake: pip install cmake)
pip install dlib
pip install -r requirements.txt
下载模型文件:

从官网下载 shape_predictor_68_face_landmarks.dat.bz2。

解压后，将 .dat 文件放置于项目根目录下的 models/ 文件夹内。

💡 使用样例
处理单张图片
以下代码展示了如何调用核心FaceAnalyzer类来处理一张图片，并获取所有分析结果。

Python

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

Bash

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
