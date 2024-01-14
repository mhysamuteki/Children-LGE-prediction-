import base64
import json
import os
import glob
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils


def process_json_files(json_dir: str, jpgs_path: str, pngs_path: str) -> None:
    json_files = glob.glob(os.path.join(json_dir, '*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')

        img = utils.img_b64_to_arr(imageData)
        label_name_to_value = {'_background_': 0, 'inside': 1, 'out': 2}
        lbl = np.zeros(img.shape[:2], dtype=np.uint8)

        masks = []
        for shape in data['shapes']:
            label_name = shape['label']
            mask = utils.shape_to_mask(img.shape[:2], shape['points'], shape_type=shape.get('shape_type'))
            label_value = label_name_to_value[label_name]

            if label_name == 'inside' or label_name == 'out':
                lbl[mask] = label_value
            masks.append(mask)

        intersection_mask = np.logical_and(masks[0], masks[1])
        lbl[intersection_mask] = label_name_to_value['_background_']

        file_name = os.path.splitext(os.path.basename(json_file))[0]
        jpg_file_path = osp.join(jpgs_path, file_name + '.jpg')
        png_file_path = osp.join(pngs_path, file_name + '.png')

        PIL.Image.fromarray(img).save(jpg_file_path)
        utils.lblsave(png_file_path, lbl)

        print(f"已保存 {file_name}.jpg 和 {file_name}.png")


if __name__ == '__main__':
    json_dir = "./datasets/before"
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"

    process_json_files(json_dir, jpgs_path, pngs_path)
