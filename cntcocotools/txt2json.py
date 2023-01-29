import json
import re
import os

import contextlib
with contextlib.suppress(Exception):
    import pycocotools.mask as mask
from mmdet.core.evaluation import coco_classes


def ann2rle(ann, h, w):
    ''' For mask.
    usage: rle = ann2rle(ann, segmentation['size'][1], segmentation['size'][0]
    '''
    segm = ann['segmentation']
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask RLE code
        rles = mask.frPyObjects(segm, h, w)
        return mask.merge(rles)
    elif isinstance(segm['counts'], list):
        # uncompressed RLE
        return mask.frPyObjects(segm, h, w)
    else:
        # rle
        return segm

def ann2mask(self, ann):
        rle = self.ann2rle(ann)
        m = mask.decode(rle)
        return m

def rle2area(rle):
    ''' For mask.
    usage: area = rle2area(rle)
    '''
    return float(mask.area(rle))


def txt2json(dir, json_path='ann_file'):
    images = []
    annotations = []
    categories = [{'id': i, 'name': cat} for i, cat in enumerate(coco_classes())]

    txts = [name for name in os.listdir(dir) if re.match(r'^\d+\.txt$', name) is not None]
    for txt in txts:
        img_id = int(re.match(r'^(\d+)\.txt$', txt).group(1))
        img_name = ''.join(txt.split('.')[:-1])

        images.append(dict(
            file_name=img_name + '.jpg',
            id=img_id,
            width=1920,
            height=1080
        ))

        with open(os.path.join(dir, txt), "r") as f:
            for line in f:
                seg = line.strip().split(",")
                for dic in categories:
                    if dic['name'] == seg[-2]:
                        cls_id = dic['id']
                        break
                x1, y1, x2, y2 = [int(n) for n in seg[:4]]

                annotations.append(dict(
                    id=len(annotations),
                    image_id=img_id,
                    category_id=cls_id,
                    bbox=[x1, y1, x2-x1, y2-y1],
                    count=int(seg[-1]),
                    area=float((x2-x1)*(y2-y1)),    # 存疑
                    iscrowd=0                       # 存疑
                ))

    ann_file = dict(images=images, annotations=annotations, categories=categories)
    with open(json_path, "w") as f:
        json.dump(ann_file, f, indent=4)


if __name__ == '__main__':
    txt2json('data/Locount/annotation_train', 'train.json')
    txt2json('data/Locount/annotation_test', 'test.json')