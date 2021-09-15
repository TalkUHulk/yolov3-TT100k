import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import os
import json
from utils.image import *
# from image import *


# TT100k 151


class COCO(Dataset):
    def __init__(self, data_dir, image_size=(416, 416)):
        super(COCO, self).__init__()
        self.data_dir = data_dir

        self.annot_path = os.path.join(self.data_dir, "annotations", f'TT100K_CoCo_format_train.json')
        self.img_dir = os.path.join(self.data_dir, "train")

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.valid_ids = list(self.coco.cats.keys())
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.rand_scales = np.arange(0.6, 1.4, 0.1)
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def rand(self, a=.0, b=1.0):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, img_id, input_shape, hue=.1, sat=1.5, val=1.5, random=True):
        """实时数据增强的随机预处理"""
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        box = np.array([anno['bbox'] + [self.cat_ids[anno['category_id']]] for anno in annotations], dtype=np.float32)

        if len(box) == 0:
            box = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)

        box[:, 2:4] += box[:, :2]  # xywh to xyxy

        image = cv2.imread(img_path)
        iw, ih = image.shape[0], image.shape[1]
        h, w = input_shape

        # -----------------------------------debug---------------------------------
        # tmp = image.copy()
        # for bb in box:
        #     label = int(bb[4])
        #     tmp = cv2.rectangle(tmp, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
        #     cat_id = list(self.cat_ids.keys())[list(self.cat_ids.values()).index(label)]
        #     print(label, cat_id, self.coco.cats[cat_id])
        # cv2.imshow("src", tmp)
        # -----------------------------------debug---------------------------------

        center = np.array([iw / 2., ih / 2.], dtype=np.float32)  # center of image
        scale = max(ih, iw) * 1.0
        flipped = False
        if random:
            # 随机选择一个尺寸来训练
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(256, iw)
            h_border = get_border(256, ih)

            # 在【w_border，-w_border】内圈随机选一个中心点
            center[0] = np.random.randint(low=w_border, high=iw - w_border)
            center[1] = np.random.randint(low=h_border, high=ih - h_border)

            if np.random.random() < 0.5:
                # 水平翻转
                flipped = True
                image = image[:, ::-1, :]
                center[0] = iw - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [w, h])
        image = cv2.warpAffine(image, trans_img, (w, h))

        to_deleted = []  # 去除不在图中的框
        for bi, bbox in enumerate(box):
            if flipped:
                bbox[[0, 2]] = iw - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_img)
            bbox[2:4] = affine_transform(bbox[2:4], trans_img)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h - 1)

            box_w = bbox[2] - bbox[0]
            box_h = bbox[3] - bbox[1]

            bbox = bbox[:4][np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            if len(bbox) == 0:
                to_deleted.append(bi)
            else:
                box[bi, :4] = bbox

        # 去除不在图中的框
        if len(to_deleted):
            box = np.delete(box, to_deleted, axis=0)
        # -----------------------------------debug---------------------------------
        # for bb in box:
        #     label = int(bb[4])
        #     image = cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 0, 255), 2)
        #     cat_id = list(self.cat_ids.keys())[list(self.cat_ids.values()).index(label)]
        #     print(label, cat_id, self.coco.cats[cat_id])
        #
        # cv2.imshow("center", image)
        # -----------------------------------debug---------------------------------
        if random:
            # 色域变换
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_BGR2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv2.cvtColor(x, cv2.COLOR_HSV2BGR) * 255

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, box

    def __getitem__(self, index):
        img_id = self.images[index]

        img, y = self.get_random_data(img_id, self.image_size[0:2])

        # -----------------------------------debug---------------------------------
        # for bb in y:
        #     label = int(bb[4])
        #     img = cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
        #     cat_id = list(self.cat_ids.keys())[list(self.cat_ids.values()).index(label)]
        #     print(label, cat_id, self.coco.cats[cat_id])
        # cv2.imshow("label", cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        return img_id, tmp_inp, tmp_targets


class COCOEval(COCO):
    def __init__(self, data_dir, image_size=(416, 416)):
        super(COCOEval, self).__init__(data_dir, image_size)
        self.data_dir = data_dir
        self.annot_path = os.path.join(self.data_dir, "annotations", f'TT100K_CoCo_format_test.json')
        self.img_dir = os.path.join(self.data_dir, "test")

        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.valid_ids = list(self.coco.cats.keys())
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def convert_eval_format(self, img_ids: list, detections):
        # cnt = 0
        coco_detections = []
        for img_id, detection in zip(img_ids, detections):
            img = self.coco.loadImgs(ids=[img_id])[0]
            #img_path = os.path.join(self.img_dir, img['file_name'])
            # print(img_path)
            # bgr = cv2.imread(img_path)

            w_ori = img['width']
            h_ori = img['height']

            scale = min(w_ori / self.image_size[0], h_ori / self.image_size[1])
            nw = int(self.image_size[0] * scale)
            nh = int(self.image_size[1] * scale)
            # 调整目标框坐标
            if detection is not None:
                detection[:, [0, 2]] = detection[:, [0, 2]] * nw / self.image_size[0]
                detection[:, [1, 3]] = detection[:, [1, 3]] * nh / self.image_size[1]
                for box in detection:
                    # bgr = cv2.rectangle(bgr,
                    #                     (int(box[0]), int(box[1])),
                    #                     (int(box[2]), int(box[3])),
                    #                     (0, 0, 255), 2)
                    tmp = {"image_id": int(img_id),
                           "category_id": list(self.cat_ids.keys())[list(self.cat_ids.values()).index(int(box[-1]))],
                           "bbox": [int(box[0]), int(box[1]), int(box[2]) - int(box[0]), int(box[3]) - int(box[1])],
                           "score": float(box[4] * box[5])}
                    coco_detections.append(tmp)

            # cv2.imshow(str(cnt), bgr)
            # cnt += 1
        # print(coco_dets)
        # cv2.waitKey()
        return coco_detections

    def run_eval(self, img_ids: list, detections, save_dir=None):
        detections = self.convert_eval_format(img_ids, detections)

        if save_dir is not None:
            result_json = os.path.join(save_dir, "results.json")
            json.dump(detections, open(result_json, "w"))

        coco_dets = self.coco.loadRes(detections)
        coco_eval = COCOeval(self.coco, coco_dets, "bbox")
        # coco_eval.params.catIds = [1]   # only test person
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats

    def __getitem__(self, index):
        img_id = self.images[index]

        img, y = self.get_random_data(img_id, self.image_size[0:2], random=False)

        # -----------------------------------debug---------------------------------
        # for bb in y:
        #     label = int(bb[4])
        #     img = cv2.rectangle(img, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (255, 0, 0), 2)
        #     cat_id = list(self.cat_ids.keys())[list(self.cat_ids.values()).index(label)]
        #     print(label, cat_id, self.coco.cats[cat_id])
        # cv2.imshow("label", cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.image_size[1]
            boxes[:, 1] = boxes[:, 1] / self.image_size[0]
            boxes[:, 2] = boxes[:, 2] / self.image_size[1]
            boxes[:, 3] = boxes[:, 3] / self.image_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_targets = np.array(y, dtype=np.float32)
        return img_id, tmp_inp, tmp_targets


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    img_ids = []
    for img_id, img, box in batch:
        images.append(img)
        bboxes.append(box)
        img_ids.append(img_id)
    images = np.array(images)
    # print("bboxes:", len(bboxes), bboxes[0].shape, bboxes[1].shape)
    return img_ids, images, bboxes


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # train_dataset = COCO("/Users/hulk/Documents/DataSets/TT100k", (608, 608))
    train_dataset = COCOEval("/Users/hulk/Documents/DataSets/TT100k", (608, 608))
    print(len(train_dataset))
    gen = DataLoader(train_dataset, shuffle=False, batch_size=2, collate_fn=yolo_dataset_collate)
    for iteration, batch in enumerate(gen):
        img_ids, images, targets = batch[0], batch[1], batch[2]
        # print(img_ids)
        # print(images.shape, targets)
