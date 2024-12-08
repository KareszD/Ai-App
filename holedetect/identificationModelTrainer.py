from ultralytics import YOLO
from ultralytics.data.utils import autosplit
import Tiff_to_JPG
import cv2
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections
from tqdm import tqdm
from patched_yolo_infer import visualize_results
from ray import tune
import glob
from pathlib import Path
import matplotlib.pyplot as plt


class YOLOTrainer():
    def __init__(self, imgsz, name, batch=16, epoch=200):
        self.imgsz = imgsz
        self.batch_size = batch
        self.epoch = epoch
        self.model_name = name
        self.project = "FoldiKutya"
        self.model = None

    def train_model(self):
        self.model = YOLO('yolov9c.pt')

        #autosplit("tmp/images", weights=(0.9, 0.1, 0.0), annotated_only=True)

        results = self.model.train(
            data='kutya.yaml',
            single_cls=True,
            patience=200,
            epochs=self.epoch,
            imgsz=self.imgsz,
            batch=self.batch_size,
            workers=4,
            deterministic=True,
            seed=42,
            name=self.model_name,
            project=self.project,
            pretrained=True,
            augment=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            cos_lr=True,
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            box=7.5,  # IOU alapú box coordináták
            cls=0.5,  # correct classification | insignificant because I only have one object
            dfl=1.5,  # osztályok közti különbség
            dropout=0.0,
            hsv_h=0.015,
            hsv_s=0.2,
            hsv_v=0.1,
            degrees=5.0,
            translate=0.05,
            scale=0.15,
            shear=2.0,
            perspective=0.0,
            flipud=0.1,
            fliplr=0.25,
            bgr=0.0,
            mosaic=0.0,  #0.5,
            mixup=0.0,  #0.5,
            copy_paste=0.0,  #0.3,
            crop_fraction=0.0,
            erasing=0.1
        )

        # results = self.model.tune(
        #     data='kutya.yaml',
        #     single_cls=True,
        #     # optimizer='SGD',
        #     # lr0=0.01,
        #     # lrf=0.0001,
        #     # momentum=0.937,
        #     # weight_decay=0.0005,
        #     # dropout=0.1,
        #     epochs=self.epoch,
        #     imgsz=self.imgsz,
        #     batch=self.batch_size,
        #     workers=0,
        #     name=self.model_name,
        #     project=self.project,
        #     pretrained=True,
        #     augment=True,
        #     plots=True, save=False, val=False, iterations=300
        # )

    def validate(self, model_name:str, data_yaml: str = "test.yaml", ):
        if self.model is None:
            self.model = YOLO(f"{model_name}/weights/best.pt")


        self.model.val(data=data_yaml,
                       imgsz=self.imgsz,
                       batch=self.batch_size,
                       device="cuda:0",
                       split="test",
                       conf=0.3,
                       iou=0.6,
                       workers=0)

        # with open(self.main_path+"\\autosplit_train.txt") as f:
        #     images_path = [line for line in f]
        #
        # for image in images_path:
        #     img = cv2.imread(image)
        #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        #     element_crops = MakeCropsDetectThem(
        #         image=img,
        #         # model_path=f"{model_name}/weights/best.pt",
        #         model=self.model,
        #         imgsz=64,
        #         segment=False,
        #         shape_x=64,
        #         shape_y=64,
        #         overlap_x=25,
        #         overlap_y=25,
        #         conf=0.7,
        #         iou=0.6,
        #         resize_initial_size=True,
        #         # show_crops=True
        #     )
        #
        #     result = CombineDetections(element_crops, nms_threshold=0.25, match_metric='IOU')
        #     box_centers = []
        #     for box in result.filtered_boxes:
        #         y_offset = result.image.shape[0]
        #         left = box[0]
        #         top = y_offset - box[1]
        #         right = box[2]
        #         bottom = y_offset - box[3]
        #         x_center = (left + right) / 2
        #         y_center = (top + bottom) / 2
        #         box_centers.append((x_center, y_center))



    def predict(self, model_name: str, outputPath: str, images: list[str], smallImgsz: int = 64):
        if self.model is None:
            self.model = YOLO(f"{model_name}/weights/best.pt")

        images_boxes = []
        #converter = Tiff_to_JPG.Converter(inputDir="out/images", outputDir='out\images', splitSize=(64, 64), input_labels=labels, is_inmemory=True)

        for image in tqdm(images, desc="Processing images"):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            element_crops = MakeCropsDetectThem(
                image=img,
                #model_path=f"{model_name}/weights/best.pt",
                model=self.model,
                imgsz=smallImgsz,
                segment=False,
                shape_x=smallImgsz,
                shape_y=smallImgsz,
                overlap_x=25,
                overlap_y=25,
                conf=0.7,
                iou=0.6,
                resize_initial_size=True,
                #show_crops=True
            )

            result = CombineDetections(element_crops, nms_threshold=0.25, match_metric='IOU')
            box_centers = []
            for box in result.filtered_boxes:
                y_offset = result.image.shape[0]
                left = box[0]
                top = y_offset - box[1]
                right = box[2]
                bottom = y_offset - box[3]
                x_center = (left + right) / 2
                y_center = (top + bottom) / 2
                box_centers.append((x_center, y_center))

            images_boxes.append((image, box_centers))


        # for image in images:
        #     data_containers = converter.convert(convert_file_name=image, isPopulate=False, isTraining=False, isUseBackGround=False)
        #     for data_container in data_containers:
        #         split_image = data_container.data
        #         split_image = cv2.cvtColor(split_image, cv2.COLOR_BGRA2RGB)
        #         results = self.model(source=split_image, stream=False, predictor=None, iou=0.5, conf=0.5, imgsz=64)
        #         for result in results:
        #             box_centers = []
        #             y_offset = result.orig_shape[0]
        #             for box in result.boxes:
        #                 left = box.xyxy.data[0][0].item()
        #                 top = y_offset - box.xyxy.data[0][1].item()
        #                 right = box.xyxy.data[0][2].item()
        #                 bottom = y_offset - box.xyxy.data[0][3].item()
        #                 x_center = (left + right) / 2
        #                 y_center = (top + bottom) / 2
        #                 x_center = self.repoint_point(x_center, result.orig_shape[1], data_container.x)
        #                 y_center = self.repoint_point(y_center, result.orig_shape[0], data_container.y)
        #                 box_centers.append((x_center, y_center))
        #                 images_boxes.append((data_container, box_centers))
        #         # result.save(filename=f"Results/{result.path.split('/')[-1][-4]}/result.png")

        return images_boxes

    def repoint_point(self, point, shape, i):
        if shape == 64:
            return point + i * 64
        else:
            p = point + i * 64
            p += shape
            return p
