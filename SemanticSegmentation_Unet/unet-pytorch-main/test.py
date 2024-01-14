import base64
import json 
import os 
import os.path as osp

import numpy as np
import PIL.Image
from labelme import utils

if __name__ == '__main__':
    jpgs_path = "datasets/JPEGImages"
    pngs_path = "datasets/SegmentationClass"

    count = os.listdir("./datasets/before/")
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
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

                if label_name == 'inside':
                    lbl[mask] = label_value
                elif label_name == 'out':
                    lbl[mask] = label_value
                masks.append(mask)

            intersection_mask = np.logical_and(masks[0], masks[1])
            lbl[intersection_mask] = label_name_to_value['_background_']

            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0] + '.jpg'))
            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0] + '.png'), lbl)
            
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
