__author__ = 'tsungyi', 'SakiRinn'

import numpy as np
import datetime
import time
from collections import defaultdict
from . import mask as maskUtils
from mmdet.models.losses import single_cnt_accuracy
import copy

class COCOeval:

    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm', counts=0):
        if not iouType:
            print('iouType not specified. use default iouType segm')
        self.cocoGt   = cocoGt
        self.cocoDt   = cocoDt
        self.evalImgs = defaultdict(list)
        self.eval     = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.params = Params(iouType=iouType)
        self._paramsEval = {}
        self.stats = []
        self.ious = {}
        if not cocoGt is None:
            self.params.imgIds = sorted(cocoGt.getImgIds())
            self.params.catIds = sorted(cocoGt.getCatIds())
            self.params.cntIds = sorted(cocoGt.getCntIds(counts))

    def _prepare(self):
        def _toMask(anns, coco):
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params

        kwargs = dict(imgIds=p.imgIds)
        if p.useCats:
            kwargs.update(catIds=p.catIds)
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(**kwargs))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(**kwargs))

        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)
        self.eval     = {}

    def evaluate(self):
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        if p.useCnts:
            p.cntIds = list(np.unique(p.cntIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p

        self._prepare()
        catIds = p.catIds if p.useCats else [-1]
        cntIds = p.cntIds if p.useCnts else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        self.ious = {(imgId, catId): computeIoU(imgId, catId)
                     for imgId in p.imgIds
                     for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats and p.useCnts:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for catId in p.catIds for _ in self._gts[imgId, catId]]
            dt = [_ for catId in p.catIds for _ in self._dts[imgId, catId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        # Sort by score (cls) and keep `maxDet` bboxes.
        inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def computeOks(self, imgId, catId):
        p = self.params
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        # Sort by score (cls).
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]

        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        if len(gts) == 0 or len(dts) == 0:
            return []

        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)

        for j, gt in enumerate(gts):
            g = np.array(gt['keypoints'])
            xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt['bbox']
            x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints'])
                xd = d[0::3]; yd = d[1::3]
                if k1>0:
                    dx = xd - xg
                    dy = yd - yg
                else:
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0)+np.max((z, xd - x1), axis=0)
                    dy = np.max((z, y0 - yd), axis=0)+np.max((z, yd - y1), axis=0)
                e = (dx**2 + dy**2) / vars / (gt['area'] + np.spacing(1)) / 2
                if k1 > 0:
                    e=e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        p = self.params
        if p.useCats and p.useCnts:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for catId in p.catIds for _ in self._gts[imgId, catId]]
            dt = [_ for catId in p.catIds for _ in self._dts[imgId, catId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]

        ious = self.ious[imgId, catId][:, gtind] \
               if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T  = len(p.iouThrs)
        CT = len(p.acThrs)
        G  = len(gt)
        D  = len(dt)
        gtm  = np.zeros((T, CT, G))
        dtm  = np.zeros((T, CT, D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T, CT, D))

        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for ctind, ct in enumerate(p.acThrs):
                    for dind, d in enumerate(dt):

                        iou = min([t , 1 - 1e-10])
                        ac  = min([ct, 1 - 1e-10])
                        m   = -1
                        for gind, g in enumerate(gt):
                            acs = single_cnt_accuracy(d['count'], g['count'])
                            if gtm[tind, ctind, gind] > 0 and not iscrowd[gind]:
                                continue
                            if m > -1 and gtIg[m] == 0 and gtIg[gind] == 1:
                                break
                            if ious[dind, gind] < iou:
                                continue
                            if acs < ac:
                                continue
                            iou = ious[dind, gind]
                            ac  = acs
                            m   = gind
                        if m == -1:
                            continue
                        dtIg[tind, ctind, dind] = gtIg[m]
                        dtm[tind, ctind, dind]  = gt[m]['id']
                        gtm[tind, ctind, m]     = d['id']

        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt])
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.broadcast_to(a, [T, CT, len(dt)])))

        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d['score'] for d in dt],
                'dtCntScores':  [d['cnt_score'] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

    def accumulate(self, p = None):
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')

        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        CT          = len(p.acThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T, CT, R, K, A, M)) # -1 for the precision of absent categories
        recall      = -np.ones((T, CT, K, A, M))
        scores      = -np.ones((T, CT, R, K, A, M))

        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)

        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)

        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:, :, 0:maxDet] for e in E], axis=-1)[:, :, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, :, 0:maxDet]  for e in E], axis=-1)[:, :, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)

                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm , np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
                    tp_sum = np.cumsum(tps, axis=-1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=-1).astype(dtype=np.float)

                    assert tp_sum.shape[:-1] == fp_sum.shape[:-1]
                    for t in range(tp_sum.shape[0]):
                        for ct in range(tp_sum.shape[1]):
                            tp = np.array(tp_sum[t, ct])
                            fp = np.array(fp_sum[t, ct])
                            nd = len(tp)
                            rc = tp / npig
                            pr = tp / (fp + tp + np.spacing(1))
                            q  = np.zeros((R, ))
                            ss = np.zeros((R, ))

                            if nd:
                                recall[t, ct, k, a, m] = rc[-1]
                            else:
                                recall[t, ct, k, a, m] = 0

                            pr = pr.tolist(); q = q.tolist()

                            for i in range(nd - 1, 0, -1):
                                if pr[i] > pr[i - 1]:
                                    pr[i - 1] = pr[i]

                            inds = np.searchsorted(rc, p.recThrs, side='left')
                            try:
                                for ri, pi in enumerate(inds):
                                    q[ri] = pr[pi]
                                    ss[ri] = dtScoresSorted[pi]
                            except:
                                pass
                            precision[t, ct, :, k, a, m] = np.array(q)
                            scores[t, ct, :, k, a, m] = np.array(ss)

        self.eval = {
            'params': p,
            'counts': [T, CT, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        def _summarize(ap=1, iouThr=None, acThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | AC={:<9} | area={:>6s} | maxDets={:>4d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(acThr)
            acStr  = '{:0.2f}:{:0.2f}'.format(p.acThrs[0], p.acThrs[-1]) \
                if acThr  is None else '{:0.2f}'.format(acThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                s = self.eval['precision']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t, ...]
                if acThr is not None:
                    ct = np.where(acThr == p.acThrs)[0]
                    s = s[:, ct, ...]
                s = s[:, :, :, :, aind, mind]
            else:
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            if len(s[s > -1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, acStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((13,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, maxDets=150)
            stats[2] = _summarize(1, iouThr=.5, acThr=.5, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, iouThr=.75, acThr=.75, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[9] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[12] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self):
        self.summarize()

class Params:

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        self.cntIds = []
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.acThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)    # ADD
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0**2, 1e5**2], [0**2, 150**2], [150**2, 300**2], [300** 2, 1e5**2]]        # EDIT
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.cntRng = [[0, 1e5], [0, 1], [2, 10], [11, 1e5]]                                        # ADD
        self.cntRngLbl = ['all', 'individual', 'medium', 'large']                                   # ADD
        self.useCats = 1
        self.useCnts = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        self.cntIds = []
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.acThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)    # ADD
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0**2, 1e5**2], [0**2, 150**2], [150**2, 300**2], [300** 2, 1e5**2]]        # EDIT
        self.areaRngLbl = ['all', 'medium', 'large']
        self.cntRng = [[0, 1e5], [0, 1], [2, 10], [11, 1e5]]                                        # ADD
        self.cntRngLbl = ['all', 'individual', 'medium', 'large']                                   # ADD
        self.useCats = 1
        self.useCnts = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        self.useSegm = None
