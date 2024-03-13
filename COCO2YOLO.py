import json
import os
import argparse

parser = argparse.ArgumentParser(description='Test yolo data.')
parser.add_argument('-j', help='JSON file', dest='json', required=True)
parser.add_argument('-o', help='path to output folder', dest='out', required=True)

args = parser.parse_args()

json_file = args.json
output = args.out


class COCO2YOLO:
    # 初始化了类的属性，加载了 JSON 文件内容，构建了 COCO 类别 ID 到名称的映射字典，打印了数据集的基本信息。
    def __init__(self):
        self._check_file_and_dir(json_file, output)
        self.labels = json.load(open(json_file, 'r', encoding='utf-8'))
        self.coco_id_name_map = self._categories()
        self.coco_name_list = list(self.coco_id_name_map.values())
        print("total images", len(self.labels['images']))
        print("total categories", len(self.labels['categories']))
        print("total labels", len(self.labels['annotations']))

    # 用于检查文件和目录是否存在，如果不存在则创建。
    def _check_file_and_dir(self, file_path, dir_path):
        if not os.path.exists(file_path):
            raise ValueError("file not found")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 构建 COCO 类别 ID 到名称的映射字典
    def _categories(self):
        categories = {}
        for cls in self.labels['categories']:
            categories[cls['id']] = cls['name']
        return categories

    # 加载图像信息，包括图像文件名、宽度和高度。
    def _load_images_info(self):
        images_info = {}
        for image in self.labels['images']:
            id = image['id']
            file_name = image['file_name']
            if file_name.find('\\') > -1:
                file_name = file_name[file_name.index('\\') + 1:]
            w = image['width']
            h = image['height']
            images_info[id] = (file_name, w, h)

        return images_info

    # 将 COCO 格式的边界框转换为 YOLO 格式。
    def _bbox_2_yolo(self, bbox, img_w, img_h, keypoints):
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        centerx = (bbox[0] + w) / 2  # 中心点横坐标
        centery = (bbox[1] + h) / 2  # 中心点横坐标
        dw = 1 / img_w
        dh = 1 / img_h
        centerx *= dw  # 中心点横坐标归一化
        w *= dw  # 框宽度归一化
        centery *= dh  # 中心点纵坐标归一化
        h *= dh  # 框长度归一化
        # 我添加的代码
        n = 21
        k = [0] * 63
        k[0] = keypoints[0]*dw
        k[1] = keypoints[1]*dw
        k[2] = keypoints[2]
        k[3] = keypoints[3]*dw
        k[4] = keypoints[4]*dw
        k[5] = keypoints[5]
        k[6] = keypoints[6]*dw
        k[7] = keypoints[7]*dw
        k[8] = keypoints[8]
        k[9] = keypoints[9]*dw
        k[10] = keypoints[10]*dw
        k[11] = keypoints[11]
        k[12] = keypoints[12]*dw
        k[13] = keypoints[13]*dw
        k[14] = keypoints[14]
        k[15] = keypoints[15]*dw
        k[16] = keypoints[16]*dw
        k[17] = keypoints[17]
        k[18] = keypoints[18]*dw
        k[19] = keypoints[19]*dw
        k[20] = keypoints[20]
        k[21] = keypoints[21]*dw
        k[22] = keypoints[22]*dw
        k[23] = keypoints[23]
        k[24] = keypoints[]

        return centerx, centery, w, h,

    # 将 COCO 格式的注释转换为 YOLO 格式
    def _convert_anno(self, images_info):
        anno_dict = dict()
        for anno in self.labels['annotations']:
            bbox = anno['bbox']
            image_id = anno['image_id']
            category_id = anno['category_id']
            # 我添加的
            keypoints = anno['keypoints']

            image_info = images_info.get(image_id)
            image_name = image_info[0]
            img_w = image_info[1]
            img_h = image_info[2]
            yolo_box = self._bbox_2_yolo(bbox, img_w, img_h, keypoints)

            anno_info = (image_name, category_id, yolo_box)
            anno_infos = anno_dict.get(image_id)
            if not anno_infos:
                anno_dict[image_id] = [anno_info]
            else:
                anno_infos.append(anno_info)
                anno_dict[image_id] = anno_infos
        return anno_dict

    # 保存类别名称到文件中
    def save_classes(self):
        sorted_classes = list(map(lambda x: x['name'], sorted(self.labels['categories'], key=lambda x: x['id'])))
        print('coco names', sorted_classes)
        with open('coco.names', 'w', encoding='utf-8') as f:
            for cls in sorted_classes:
                f.write(cls + '\n')
        f.close()

    def coco2yolo(self):
        print("loading image info...")
        images_info = self._load_images_info()
        print("loading done, total images", len(images_info))

        print("start converting...")
        anno_dict = self._convert_anno(images_info)
        print("converting done, total labels", len(anno_dict))

        print("saving txt file...")
        self._save_txt(anno_dict)
        print("saving done")

    def _save_txt(self, anno_dict):
        for k, v in anno_dict.items():
            file_name = os.path.splitext(v[0][0])[0] + ".txt"
            with open(os.path.join(output, file_name), 'w', encoding='utf-8') as f:
                print(k, v)
                for obj in v:
                    cat_name = self.coco_id_name_map.get(obj[1])
                    category_id = self.coco_name_list.index(cat_name)
                    box = ['{:.6f}'.format(x) for x in obj[2]]
                    box = ' '.join(box)
                    line = str(category_id) + ' ' + box
                    f.write(line + '\n')


if __name__ == '__main__':
    c2y = COCO2YOLO()
    c2y.coco2yolo()