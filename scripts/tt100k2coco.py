import os
import json
import cv2


_type = "test"

src = "DataSets/TT100k/"
anno = os.path.join(src, "annotations.json")

categories = ["__background__"]
dataset = {'categories': [], 'images': [], 'annotations': []}

with open(anno, "r") as j:
    annotations = json.load(j)

types = annotations["types"]
imgs = annotations["imgs"]

print("types num:", len(types))

# 151
cat = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'io', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pg', 'ph1.5', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm2', 'pm20', 'pm30', 'pm35', 'pm40', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'po', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w3', 'w30', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w45', 'w46', 'w47', 'w5', 'w55', 'w57', 'w58', 'w59', 'w63', 'w66', 'w8', 'wo', 'w15', 'pl0']

# for i, t in enumerate(types, 1):
#     dataset['categories'].append(
#         {'id': i, 'name': t, 'supercategory': t[0]}
#     )
#     categories.append(t)

for i, t in enumerate(cat, 1):
    dataset['categories'].append(
        {'id': i, 'name': t, 'supercategory': t[0]}
    )
    categories.append(t)


g_index = 1

for id, (img, info) in enumerate(imgs.items(), 1):
    if _type in info['path']:
        bgr = cv2.imread(os.path.join(src, info['path']))
        h, w, _ = bgr.shape
        objects = info['objects']
        dataset['images'].append({'file_name': info['path'].split("/")[-1],
                                  'id': id,
                                  'width': w,
                                  'height': h})
        for obj in objects:
            bbox = obj["bbox"]
            x1 = bbox["xmin"]
            y1 = bbox["ymin"]
            x2 = bbox["xmax"]
            y2 = bbox["ymax"]

            label = obj["category"]

            if "ellipse_org" in obj:
                segmentation = obj["ellipse_org"]
            elif "polygon" in obj:
                segmentation = obj["polygon"]
            else:
                segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if label not in categories:
                continue
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': categories.index(label),
                'id': g_index,
                'image_id': id,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': segmentation
            })
            g_index += 1

json_name = os.path.join(src, '{}.json'.format(f"TT100K_CoCo_format_151_{_type}"))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
print('done')
