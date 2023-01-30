import warnings

import cntcocotools
from cntcocotools.coco import COCO as _COCO
from cntcocotools.cocoeval import COCOeval as _COCOeval


class COCO(_COCO):

    def __init__(self, annotation_file=None):
        # if getattr(cntcocotools, '__version__', '0') >= '12.0.2':
        #     warnings.warn(
        #         'mmpycocotools is deprecated. Please install official pycocotools by "pip install pycocotools"',  # noqa: E501
        #         UserWarning)
        super().__init__(annotation_file=annotation_file)
        self.img_ann_map = self.imgToAnns
        self.cat_img_map = self.catToImgs
        self.cnt_img_map = self.cntToImgs

    def get_ann_ids(self, img_ids=[], cat_ids=[], cnt_ids=[], area_rng=[], iscrowd=None):
        return self.getAnnIds(img_ids, cat_ids, cnt_ids, area_rng, iscrowd)

    def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
        return self.getCatIds(cat_names, sup_names, cat_ids)

    def get_img_ids(self, img_ids=[], cat_ids=[]):
        return self.getImgIds(img_ids, cat_ids)

    def get_cnt_ids(self, max_count=0):
        return self.getCntIds(max_count)

    def load_anns(self, ids):
        return self.loadAnns(ids)

    def load_cats(self, ids):
        return self.loadCats(ids)

    def load_imgs(self, ids):
        return self.loadImgs(ids)


COCOeval = _COCOeval
