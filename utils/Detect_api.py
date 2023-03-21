from utils.utils import *
import cv2
from utils.config import model_file,img_size

class Yolo_inference():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type != 'cpu'    # half precision only supported on CUDA
        self.model = torch.load(model_file, map_location=self.device)['model'].float().eval()    # load FP32 model
        if self.half:
            self.model.half()    # to FP16

    def transforms(self,image):
        """
        预处理
        :param image:原图
        :return: 处理后的图片
        """
        # time4 = time.time()
        img = letterbox(image,new_shape=img_size)[0]
        img = img[:,:,::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # print(img)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def Process(self,pred,img,image,names,colors):
        image_ = image.copy()
        number_name = []
        img_crop = []
        h, w, _ = image_.shape
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                # Write results
                det = det[det[:, 0].argsort(descending=False)]

                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    img_crop.append(image_[int(xyxy[1]):int(xyxy[3]),int(xyxy[0]):int(xyxy[2])])
                    plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=3)
                    number_name.append([str(int(xyxy[0])),str(int(xyxy[1])),str(int(xyxy[2])),str(int(xyxy[3])),names[int(cls)]])
        return image, number_name,img_crop

    def detect(self,image,conf_thres=0.3, iou_thres=0.3):
        """
        前向推理
        :param image:图片
        :param model: 模型
        :param conf_thres: 置信度阈值
        :param iou_thres: iou阈值
        :return: 原图和裁剪部分
        """
        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names     # class name
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]   # box colors

        # Run inference

        img = self.transforms(image)
        with torch.no_grad():

            pred = self.model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        return self.Process(pred,img,image,names,colors)

