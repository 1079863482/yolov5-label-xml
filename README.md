# yolov5-label-xml

### 使用yolov5实现的一个预标注程序

1. 将预训练好的权重放入models/weights中，修改utils/config.py配置文件；

2. 将预标注的图片放入inference/images

3. 运行demo.py文件，结束后预标注的xml文件存放在inference/xmls中、

4. 使用labelimg进行微调。
