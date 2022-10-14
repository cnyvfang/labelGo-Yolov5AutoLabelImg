# labelGo

<p>一个基于<a href="https://github.com/tzutalin/labelImg">labelImg</a>与<a href="https://github.com/ultralytics/yolov5">YOLOv5</a>算法的半自动标注工具</p>
<p>通过现有的YOLOv5 Pytorch模型对数据集进行半自动标注</p>

## News
**现已支持最新版本YOLOv5以及classes.txt的自动生成**

## YOLOv5半自动标注功能演示
![image](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/demo/demo1.gif) 
## 一键将YOLO格式标签转换为VOC格式标签功能演示
![image](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/demo/demo2.gif) 

## 注意
<p>如果遇到问题，欢迎在issue中提出。</p>
<p>标注文件保存位置与图片文件夹相同。</p>
<p>推荐使用的Python版本：Python3.8。</p>
<p>建议使用conda环境进行操作。</p>
<p>该项目完全免费，不会且禁止进行任何商业出售。</p>
<p>该项目已支持最新版本YOLOv5，如需要使用支持YOLOv5 version5的旧版本，可以在Release中找到旧版本代码。</p>
<p>切换中文版本请编辑\libs\stringBundle.py，注释第52行，取消注释54行。（原理：":/strings"替换为":/strings-zh-CN"）</p>


## 使用方法
<p>1.从 git 中获取项目</p>

```bash
git clone https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg.git
```

<p>2.切换操作目录到工程目录</p>

```bash
cd labelGo-Yolov5AutoLabelImg
```

<p>3.配置环境</p>

```bash
pip install -r requirements.txt
```

<p>4（可选）.变更成中文版，请编辑\libs\stringBundle.py，注释第52行，取消注释54行。</p>

<p>5.启动应用程序</p>

```bash
python labelGo.py
```
<p>6.点击“打开目录”按钮选择存放图片的文件夹</p>

<p>7.点击“自动标注”按钮确认信息无误后选择已训练的yolov5 pytorch模型完成自动标注</p>

<p>8.根据实际要求，对自动标注结果进行调整后保存即可</p>

## 致谢
感谢[tangtang666](https://github.com/tangtang666)提交的针对最新版YOLOv5的支持

感谢[Iceprism](https://github.com/Iceprism)对中文版Bug的修复
