<h1>labelGo</h1>
<p>Guide Language:<a href="https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/readme_zh_cn.md">简体中文</a></p>
<p>A graphical Semi-automatic annotation tool based on <a href="https://github.com/tzutalin/labelImg">labelImg</a> and <a href="https://github.com/ultralytics/yolov5">YOLOv5</a></p>
<p>Semi-automatic annotation of datasets by existing yolov5 pytorch models</p>

## Demonstration of semi-automatic labeling function
![image](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/demo/demo1.gif) 
## Function demonstration of converting Yolo format to VOC format with one click
![image](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/demo/demo2.gif) 

## Attention
<p>If there is a problem, please put it forward in the issue</p>
<p>Please put classes.txt under the marked dataset folder in advance</p>
<p>The annotation file is saved in the same location as the picture folder</p>
<p>Recommended version of python: python 3.8</p>
<p>Recommended for conda environments</p>
<p>The item is completely free and it is forbidden to sell the item in any way. </p>
<p>Note: This project only supports Version 5 of YOLOv5 for the time being.</p>


## Installation and use
<p>1.Fetching projects from git</p>

```bash
git clone https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg.git
```

<p>2.Switching the operating directory to the project directory</p>

```bash
cd labelGo-Yolov5AutoLabelImg
```

<p>3.Installation environment</p>

```bash
pip install -r requirements.txt
```

<p>4.Modify the contents of the /data/predefined_classes.txt file in the directory to your own category</p>

<p>5.Launching applications</p>

```bash
python labelGo.py
```

<p>6. Click on the "Open directory" button to select the folder where the images are stored</p>

<p>7. Click on the "Auto Annotate" button to confirm that the information is correct and then select the trained yolov5 pytorch model to complete the auto annotation</p>

<p>8. Adjust the automatic annotation results according to the actual requirements and save them</p>
