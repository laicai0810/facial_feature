# **👨‍💻 Facial Feature Extractor \- 自动化面部特征提取与分析系统**

\<p align="center"\>  
\<a href="https://github.com/laicai0810/facial\_feature/tree/main"\>\<img src="https://img.shields.io/badge/GitHub-Repo-blue.svg" alt="GitHub Repo"\>\</a\>  
\<img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python版本"\>  
\<img src="https://img.shields.io/badge/License-MIT-green.svg" alt="许可证"\>  
\<img src="https://img.shields.io/badge/Dlib-19.24-orange.svg" alt="Dlib版本"\>  
\<img src="https://img.shields.io/badge/OpenCV-4.x-blue.svg" alt="OpenCV版本"\>  
\</p\>  
一套功能全面的Python库，用于从图像中检测人脸、提取面部关键点、执行图像增强，并计算超过150种丰富的几何面部特征。本项目基于强大的 **Dlib** 和 **OpenCV** 库构建，旨在为下游的机器学习任务（如情绪识别、生物特征分析、金融风控等）提供高维度、结构化的特征数据。

## **✨ 核心功能**

* **👨 人脸检测**: 采用Dlib的HOG特征+线性SVM分类器，高效定位正面人脸区域。  
* **📍 关键点预测**: 基于回归树集成算法，精准提取68个面部关键点。  
* **📐 面部对齐**: 利用关键点信息生成姿态归一化的标准尺寸面部图像，消除头部姿态变化带来的影响。  
* **🎨 图像增强**: 提供一套可选的图像预处理流程，包括光照归一化 (CLAHE)、灰度转换和图像降噪 (双边滤波)等。  
* **📊 特征计算**: 自动计算超过150种面部几何特征，涵盖距离、角度、比率、对称性与轮廓面积等多个维度。

## **🗺️ 核心处理流程**

下图直观地展示了单张图片输入系统后，从原始像素到最终结构化特征数据的完整处理流程。

graph TD  
    A\[🖼️ 输入原始图像\] \--\> B{1. 人脸检测};  
    B \--\> |检测到人脸| C(dlib.rectangle);  
    B \--\> |未检测到| F\[❌ 输出: 无人脸\];  
    C \--\> D{2. 关键点定位};  
    D \--\> |定位成功| E\[📍 68个面部关键点\];  
    D \--\> |定位失败| G\[❌ 输出: 关键点错误\];  
    E \--\> H{3. 面部对齐};  
    H \--\> I\[👤 标准化面部切片 (Chip)\];  
    I \--\> J{4. 图像增强 (可选)};  
    J \--\> K\[✨ 增强后的图像\];  
    E & K \--\> L{5. 几何特征计算};  
    L \--\> M\[📊 150+ 结构化特征\];  
    M \--\> N\[💾 输出: 特征数据 (CSV)\];

    subgraph "核心处理模块 (facial\_feature\_extractor)"  
        B; C; D; E; H; I; J; K; L; M;  
    end

    style A fill:\#f9f,stroke:\#333,stroke-width:2px  
    style N fill:\#ccf,stroke:\#333,stroke-width:2px

## **🔬 关键技术深度解析**

| 步骤 | 核心模块 | 技术细节 | 可视化参考 |
| :---- | :---- | :---- | :---- |
| **1\. 人脸检测** | detection.py | **技术**: Dlib内置的基于**方向梯度直方图 (HOG)** 特征和**线性支持向量机 (SVM)** 的分类器。\<br\>**流程**: 灰度转换 \-\> 构建图像金字塔 \-\> 滑动窗口计算HOG特征 \-\> SVM分类 \-\> 选择面积最大的人脸。 |  |
| **2\. 关键点定位** | landmarks.py | **技术**: Dlib的 shape\_predictor，一种基于回归树集成的快速算法。\<br\>**模型**: 使用预训练的shape\_predictor\_68\_face\_landmarks.dat模型。\<br\>**输出**: 68个(x, y)坐标点，精确勾勒出面部轮廓。 |  |
| **3\. 面部对齐** | landmarks.py | **技术**: 调用dlib.get\_face\_chip函数。\<br\>**流程**: 内部执行相似性变换（旋转、缩放、平移），将眼睛和鼻子置于标准位置，生成一张固定大小、姿态基本校正后的人脸图像，为后续特征计算提供规范化输入。 |  |
| **4\. 图像增强** | enhancement.py | **光照归一化**: 采用**限制对比度的自适应直方图均衡化 (CLAHE)** 算法，改善光照不均问题。\<br\>**降噪处理**: 使用**双边滤波器 (Bilateral Filter)**，在平滑图像的同时能够很好地保留边缘信息。 |  |
| **5\. 特征计算** | features.py | **基础**: 所有特征的计算都基于第2步得到的68个关键点的坐标。\<br\>**特征类别**: 距离（瞳孔间距）、比率（眼宽高比EAR）、角度（下颌角）、面积/周长（嘴唇轮廓）等。 |  |

## **📈 特征维度详解**

本系统共提取超过150个几何特征，以下为关键特征维度的分类与详细说明：

#### **1\. 面部轮廓与比例 (Facial Contour & Proportions)**

这些特征描述了脸部的基本形状、大小和整体比例。

* **face\_width\_max\_jaw**: 下颌最宽处距离，衡量脸部宽度。  
* **face\_height\_nose\_bridge\_to\_chin**: 脸长，从鼻梁顶部到下巴尖的垂直距离。  
* **face\_width\_to\_height\_ratio**: 脸的宽高比，是判断脸型（如圆脸、长脸）的重要指标。  
* **jawline\_length**: 下颌线轮廓的总长度。  
* **jaw\_polygon\_area**: 由下颌线和两端点连线构成的多边形面积。  
* **chin\_angle**: 由下巴两侧点与下巴尖构成的角度，反映下巴的尖锐程度。  
* **forehead\_proxy\_height**: 额头高度的代理值（眉心到鼻梁顶部）。  
* **middle\_third\_height\_glabella\_to\_subnasale**: 中庭高度（眉心到鼻底）。  
* **lower\_third\_height\_subnasale\_to\_chin**: 下庭高度（鼻底到下巴）。  
* **face\_thirds\_ratio\_\***: 上庭、中庭、下庭之间的高度比例，用于“三庭”分析。

#### **2\. 眼部特征 (Eye Features)**

精细刻画眼睛的大小、形状和位置。

* **eye\_width\_right / eye\_width\_left**: 左右眼的宽度（内外眼角距离）。  
* **eye\_vertical\_height\_right / eye\_vertical\_height\_left**: 左右眼的垂直高度。  
* **eye\_aspect\_ratio\_right / eye\_aspect\_ratio\_left (EAR)**: 经典的眼宽高比，对眨眼检测和疲劳度分析非常敏感。  
* **avg\_ear**: 左右眼EAR的平均值。  
* **inter\_ocular\_distance\_inner**: 内眼角间距。  
* **inter\_ocular\_distance\_outer**: 外眼角间距。  
* **eye\_area\_right / eye\_area\_left**: 左右眼轮廓所围成的面积。  
* **five\_eyes\_metric\_1**: 内眼角间距与平均眼宽的比值，用于“五眼”比例分析。

#### **3\. 眉毛特征 (Eyebrow Features)**

描述眉毛的长度、弯曲度、位置和倾斜状态。

* **eyebrow\_length\_right / eyebrow\_length\_left**: 左右眉毛的长度。  
* **eyebrow\_arch\_height\_right / eyebrow\_arch\_height\_left**: 眉峰相对于眉毛两端连线的高度，反映眉毛的弯曲程度。  
* **eyebrow\_tilt\_angle\_right / eyebrow\_tilt\_angle\_left**: 眉毛的倾斜角度。  
* **eyebrow\_to\_eye\_dist\_right / eyebrow\_to\_eye\_dist\_left**: 眉毛到眼睛的平均距离。

#### **4\. 鼻部特征 (Nose Features)**

量化鼻子的大小和形状。

* **nose\_length**: 鼻长（鼻梁顶部到鼻尖）。  
* **nose\_width\_nostrils**: 鼻翼宽度。  
* **nose\_length\_to\_width\_ratio**: 鼻子的长宽比。  
* **nose\_bridge\_tilt\_angle**: 鼻梁的倾斜角度。

#### **5\. 嘴部特征 (Mouth Features)**

描述嘴唇的尺寸、形状和人中区域。

* **mouth\_width\_corners**: 嘴角宽度。  
* **mouth\_height\_outer\_lips\_center**: 外唇在中心点的垂直高度。  
* **mouth\_aspect\_ratio\_outer**: 外唇的宽高比。  
* **upper\_lip\_thickness\_center**: 上唇厚度。  
* **lower\_lip\_thickness\_center**: 下唇厚度。  
* **philtrum\_length**: 人中长度。

#### **6\. 情绪与微表情几何线索 (Emotion & Micro-expression Cues)**

这些特征旨在捕捉与情绪相关的细微面部肌肉变化，对于欺诈识别等场景尤为关键。

* **tension\_eyebrow\_gap\_horizontal\_dist**: 眉间水平距离，紧张或皱眉时通常会缩小。  
* **tension\_lip\_press\_ratio**: 嘴唇紧绷度（垂直距离/水平距离），抿嘴时该比值会变小。  
* **tension\_jaw\_clench\_metric**: 下巴收紧度代理值，通过下颌角到下巴尖的距离衡量。  
* **brow\_lower\_intensity\_y\_diff**: 皱眉强度，通过内外眉毛的垂直高度差计算。  
* **anger\_lip\_corner\_pull\_down\_avg\_y**: 嘴角下拉程度，与愤怒或悲伤情绪相关。  
* **smile\_lip\_corner\_pull\_up\_avg\_y**: 嘴角上扬程度，微笑的关键指标。  
* **smile\_cheek\_raise\_proxy\_\***: 脸颊肌肉隆起代理值，衡量“苹果肌”的上抬，是真实微笑（杜胥内微笑）的重要组成部分。

## **🚀 安装指南**

1. **克隆仓库:**  
   git clone https://github.com/laicai0810/facial\_feature.git  
   cd facial\_feature

2. **创建并激活Python虚拟环境:**  
   python \-m venv venv  
   source venv/bin/activate  \# Windows: venv\\Scripts\\activate

3. 安装依赖:  
   强烈建议先单独安装dlib，因为它可能需要编译环境。  
   \# (可能需要先安装 cmake: pip install cmake)  
   pip install dlib  
   pip install \-r requirements.txt

4. **下载模型文件:**  
   * 从[官网](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)下载 shape\_predictor\_68\_face\_landmarks.dat.bz2。  
   * 解压后，将 .dat 文件放置于项目根目录下的 models/ 文件夹内。

## **⚙️ 使用样例**

### **处理单张图片**

以下代码展示了如何调用核心FaceAnalyzer类来处理一张图片，并获取所有分析结果。

from facial\_feature\_extractor.analysis import FaceAnalyzer  
from facial\_feature\_extractor.utils import save\_image, draw\_landmarks\_on\_image  
import pprint

\# 1\. 初始化分析器，指定dlib模型路径  
analyzer \= FaceAnalyzer(shape\_predictor\_path='models/shape\_predictor\_68\_face\_landmarks.dat')

\# 2\. 调用核心方法处理图片  
image\_path \= 'path/to/your/image.jpg' \# \<-- 替换为你的图片路径  
result \= analyzer.process\_image(image\_path)

\# 3\. 查看和使用结果  
print(f"处理状态: {result\['status'\]}")  
if result\['status'\] \== 'success':  
    print(f"检测到的人脸面积: {result\['face\_area'\]}")

    \# 打印部分计算出的特征  
    print("\\n--- 部分特征展示 \---")  
    features\_to\_show \= {k: v for k, v in result\['features'\].items() if 'eye\_aspect' in k or 'mouth' in k}  
    pprint.pprint(features\_to\_show)

    \# 在最终处理后的图像上绘制关键点并保存  
    if result\['final\_image'\] is not None:  
        final\_img\_with\_landmarks \= draw\_landmarks\_on\_image(result\['final\_image'\], result\['landmarks'\])  
        save\_image(final\_img\_with\_landmarks, 'output\_with\_landmarks.png')  
        print("\\n已保存带关键点的处理后图像至 'output\_with\_landmarks.png'")

### **批量处理图片**

项目scripts/目录下提供了2\_run\_batch\_processing.py脚本，用于处理整个文件夹的图片并将所有结果汇总到一个CSV文件中。

1. 将待处理的图片放入指定文件夹（如 data/input\_images/）。  
2. 根据需要修改脚本顶部的配置变量。  
3. 执行脚本:  
   python scripts/2\_run\_batch\_processing.py

## **📁 项目结构**

.  
├── facial\_feature\_extractor/   \# 核心库代码  
│   ├── \_\_init\_\_.py  
│   ├── analysis.py             \# 主分析器类  
│   ├── detection.py            \# 人脸检测模块  
│   ├── enhancement.py          \# 图像增强模块  
│   ├── features.py             \# 特征计算模块  
│   ├── landmarks.py            \# 关键点定位与对齐模块  
│   └── utils.py                \# 工具函数  
├── scripts/                    \# 示例脚本  
│   ├── 1\_download\_images.py  
│   └── 2\_run\_batch\_processing.py  
├── models/                     \# 存放dlib模型文件  
│   └── shape\_predictor\_68\_face\_landmarks.dat  
├── .gitignore  
├── README.md  
└── requirements.txt

## **📜 许可证**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源许可证。
