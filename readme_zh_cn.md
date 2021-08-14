<h1>labelGo</h1>
<p>一个基于<a href="https://github.com/tzutalin/labelImg">labelImg</a>与Yolov5算法的半自动标注工具</p>
<p>通过现有的yolov5 pytorch模型对数据集进行半自动标注</p>

![image](https://github.com/cnyvfang/labelGo-Yolov5AutoLabelImg/blob/master/demo/demo1.gif) 

## 注意
<p>如果遇到问题，欢迎在issue中提出</p>
<p>请提前将classes.txt放置于被标注数据集文件夹下</p>

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

<p>4.启动应用程序</p>

```bash
python labelGo.py
```
