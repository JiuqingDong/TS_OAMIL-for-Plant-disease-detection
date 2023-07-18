import itertools
import logging
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from .custom import CustomDataset

import torch
from mmcv.parallel import DataContainer as DC

try:
    import pycocotools

    assert pycocotools.__version__ >= '12.0.2'
except AssertionError:
    raise AssertionError('Incompatible version of pycocotools is installed. '
                         'Run pip uninstall pycocotools first. Then run pip '
                         'install mmpycocotools to install open-mmlab forked '
                         'pycocotools.')


@DATASETS.register_module()
class CocoDataset(CustomDataset):
    CLASSES = ('blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease')  # , 'background', 'healthy', 'unknown')
    # CLASSES = ('wheat',)
    # CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self, **kwargs):
        super(CocoDataset, self).__init__(**kwargs)
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()


    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        # load noisy annotations in training
        if not self.test_mode:
            box_noise_level = self.kwargs['box_noise_level']
            ann_name = ann_file.split('/')[-1].split('.')[0]
            prefix = './data/GWHD/noisy_pkl/'
            if box_noise_level > 0:
                ann_file = '{}{}_noise-r{:.1f}.pkl'.format(prefix, ann_name, box_noise_level)
            data_infos = mmcv.load(ann_file)
            return data_infos
        else:
            self.coco = COCO(ann_file)
            self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
            self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
            self.img_ids = self.coco.get_img_ids()
            data_infos = []

            for i in self.img_ids:
                info = self.coco.load_imgs([i])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
            return data_infos

    def get_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        if self.test_mode:
            img_id = self.data_infos[idx]['id']
            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            ann_info = self.coco.load_anns(ann_ids)
            return self._parse_ann_info(self.data_infos[idx], ann_info)
        else:
            return self.data_infos[idx]['ann']

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def format_extra(self, data):
        data['img_metas'].data['gt_bboxes'] = data['gt_bboxes'].data

    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            self.format_extra(data)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and img_info['ann']['bboxes'].shape[0] == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_subset_by_classes(self):
        """Get img ids that contain any category in class_ids.

        Different from the coco.getImgIds(), this function returns the id if
        the img contains one of the categories rather than all.

        Args:
            class_ids (list[int]): list of category ids

        Return:
            ids (list[int]): integer list of img ids
        """

        ids = set()
        for i, class_id in enumerate(self.cat_ids):
            ids |= set(self.coco.cat_img_map[class_id])
        self.img_ids = list(ids)

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
        return data_infos

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        image_id = []
        json_results_mid = dict()
        json_results_post_process = []
        for i in json_results:
            category = i['image_id']
            image_id.append(category)
        #
        image_id = list(set(image_id))
        for img in image_id:
            json_results_mid[str(img)] = []
            for i in json_results:
                if i['image_id'] == img:
                    bbox = [i['bbox']]
                    score = i['score']
                    bbox.append(score)
                    cid = i['category_id']
                    bbox.append(cid)
                    json_results_mid[str(img)].append(bbox)
        #
        for i in json_results_mid:
            tmp_list = process_result(i, json_results_mid[i])
            if tmp_list is not None and len(tmp_list) != 0:
                for ob in tmp_list:
                    json_results_post_process.append(ob)

        return json_results
        # return json_results_post_process

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    # set classwise=true and iou_thrs=[0.5] to calculate the mAP50.

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs= None, #,  # None,#
                 metric_items=None):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        eval_results = {}
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                cocoDt = cocoGt.loadRes(result_files[metric])
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            iou_type = 'bbox' if metric == 'proposal' else metric
            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results


def transfer(category):
    if category == 1:
        new_category = "blossom_end_rot"
    elif category == 2:
        new_category = "graymold"
    elif category == 3:
        new_category = "powdery_mildew"
    elif category == 4:
        new_category = "spider_mite"
    elif category == 5:
        new_category = "spotting_disease"
    elif category == 6:
        new_category = "background"
    elif category == 7:
        new_category = "healthy"
    elif category == 8:
        new_category = "unknown"
    else:
        raise EOFError
    return new_category


def transfer_gwhd(category):
    if category == 1:
        new_category = "wheat"

    else:
        raise EOFError
    return new_category


def xywh2xyxy(x, y, w, h):
    xmin = x
    ymin = y
    xmax = x + w
    ymax = y + h
    return xmin, ymin, xmax, ymax


def back(xmin, ymin, xmax, ymax):
    x = xmin
    y = ymin
    w = xmax - xmin
    h = ymax - ymin
    return x, y, w, h


def get_new_value(object):
    x = object[0][0]
    y = object[0][1]
    w = object[0][2]
    h = object[0][3]
    xmin, ymin, xmax, ymax = xywh2xyxy(x, y, w, h)
    score = object[1]
    category = object[2]
    new_category = transfer(category)

    return new_category, xmin, ymin, xmax, ymax, score


def get_new_value_gwhd(object):
    x = object[0][0]
    y = object[0][1]
    w = object[0][2]
    h = object[0][3]
    xmin, ymin, xmax, ymax = xywh2xyxy(x, y, w, h)
    score = object[1]
    category = object[2]
    new_category = transfer_gwhd(category)

    return new_category, xmin, ymin, xmax, ymax, score


def my_IOU(rec_1, rec_2):
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])  # 第一个bbox面积 = 长×宽
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])  # 第二个bbox面积 = 长×宽
    sum_s = s_rec1 + s_rec2  # 总面积
    left = max(rec_1[0], rec_2[0])  # 并集左上角顶点横坐标
    right = min(rec_1[2], rec_2[2])  # 并集右下角顶点横坐标
    bottom = max(rec_1[1], rec_2[1])  # 并集左上角顶点纵坐标
    top = min(rec_1[3], rec_2[3])  # 并集右下角顶点纵坐标
    if left >= right or top <= bottom:  # 不存在并集的情况
        return 0, s_rec1, s_rec2
    else:
        inter = (right - left) * (top - bottom)  # 求并集面积
        iou = (inter / (sum_s - inter)) * 1.0  # 计算IOU
        # Io1 = inter/ s_rec1
        # Io2 = inter/ s_rec2
    return iou, s_rec1, s_rec2


def selected_bbox(rec_1, rec_2, c1, c2, score_1, score_2):
    if score_1 <= 0.75:
        return False
    iou, s_rec1, s_rec2 = my_IOU(rec_1, rec_2)
    if c1 == c2:
        if iou <= 0.20:
            return True
        # elif 0.1<iou<0.25:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     else:
        #         return True

        elif iou > 0.20:
            if s_rec1 >= s_rec2:
                return True
            else:
                return False
        else:
            raise EOFError

    elif c1 != c2:
        if iou < 0.5:
            return True

        # elif 0.25 <= iou <= 0.5:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     elif s_rec1 > s_rec2:
        #         return True
        #     else:
        #         return False
        elif iou > 0.5:
            if score_1 > score_2:
                return True
            elif score_1 < score_2:
                return False
            else:
                if s_rec1 >= s_rec2:
                    return True
                else:
                    return False
        else:
            raise EOFError


def selected_bbox_gwhd(rec_1, rec_2, c1, c2, score_1, score_2):
    if score_1 <= 0.8:
        return False
    iou, s_rec1, s_rec2 = my_IOU(rec_1, rec_2)
    if c1 == c2:
        if iou <= 0.50:
            return True
        # elif 0.1<iou<0.25:
        #     if rec_1[0] < rec_2[0] and rec_1[1] < rec_2[1] and rec_1[2] > rec_2[2] and rec_1[3] > rec_2[3]:
        #         return True
        #     elif rec_2[0] < rec_1[0] and rec_2[1] < rec_1[1] and rec_2[2] > rec_1[2] and rec_2[3] > rec_1[3]:
        #         return False
        #     else:
        #         return True

        elif iou > 0.50:
            if s_rec1 >= s_rec2:
                return True
            else:
                return False
        else:
            raise EOFError


def process_result(i, object_list):
    temp_list = []
    if len(object_list) == 0:
        return None

    if len(object_list) == 1:
        obj = object_list[0]
        classes, xmin, ymin, xmax, ymax, score = get_new_value(obj)
        if classes not in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
            return None
        else:
            x, y, w, h = back(xmin, ymin, xmax, ymax)
            info = {'image_id': int(i), 'bbox': [x, y, w, h], 'score': score, 'category_id': obj[2]}
            temp_list.append(info)
            return temp_list

    for n1 in range(len(object_list)):
        classes_1, xmin_1, ymin_1, xmax_1, ymax_1, score_1 = get_new_value(object_list[n1])
        if classes_1 in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
            rec_1 = [xmin_1, ymin_1, xmax_1, ymax_1]
            Save = True

            for n2 in range(len(object_list)):
                classes_2, xmin_2, ymin_2, xmax_2, ymax_2, score_2 = get_new_value(object_list[n2])

                if n1 != n2 and classes_2 in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite',
                                              'spotting_disease']:
                    rec_2 = [xmin_2, ymin_2, xmax_2, ymax_2]
                    Save = selected_bbox(rec_1, rec_2, classes_1, classes_2, score_1, score_2)

                if Save is False:
                    break

            if Save is True:
                x, y, w, h = back(xmin_1, ymin_1, xmax_1, ymax_1)

                info = {'image_id': int(i), 'bbox': [x, y, w, h], 'score': score_1, 'category_id': object_list[n1][2]}
                temp_list.append(info)

    return temp_list


def process_result_gwhd(i, object_list):
    temp_list = []
    if len(object_list) == 0:
        return None

    if len(object_list) == 1:
        obj = object_list[0]
        classes, xmin, ymin, xmax, ymax, score = get_new_value_gwhd(obj)
        if classes not in ['wheat',]:
            return None
        else:
            x, y, w, h = back(xmin, ymin, xmax, ymax)
            info = {'image_id': int(i), 'bbox': [x, y, w, h], 'score': score, 'category_id': obj[2]}
            temp_list.append(info)
            return temp_list

    for n1 in range(len(object_list)):
        classes_1, xmin_1, ymin_1, xmax_1, ymax_1, score_1 = get_new_value_gwhd(object_list[n1])
        if classes_1 in ['wheat',]:
            rec_1 = [xmin_1, ymin_1, xmax_1, ymax_1]
            Save = True

            for n2 in range(len(object_list)):
                classes_2, xmin_2, ymin_2, xmax_2, ymax_2, score_2 = get_new_value_gwhd(object_list[n2])

                if n1 != n2 and classes_2 in ['wheat',]:
                    rec_2 = [xmin_2, ymin_2, xmax_2, ymax_2]
                    Save = selected_bbox_gwhd(rec_1, rec_2, classes_1, classes_2, score_1, score_2)

                if Save is False:
                    break

            if Save is True and xmin_1<(xmax_1-2) and ymin_1<(ymax_1-2):
                x, y, w, h = back(xmin_1, ymin_1, xmax_1, ymax_1)

                info = {'image_id': int(i), 'bbox': [x, y, w, h], 'score': score_1, 'category_id': object_list[n1][2]}
                temp_list.append(info)

    return temp_list