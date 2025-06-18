
import json
import argparse

parser = argparse.ArgumentParser(description="Evaluate VLM predictions")
parser.add_argument("--dataset", default="kvasir_valid-qwen-6task-test.jsonl",
                    help="Path to ground truth JSONL file")
parser.add_argument("--results", default="kvasir_valid-qwen-6task-test-result.jsonl",
                    help="Path to prediction results JSONL file")
args = parser.parse_args()

val_dataset = [json.loads(r) for r in open(args.dataset).readlines()]
val_results = [json.loads(r) for r in open(args.results).readlines()]
# all tasks  {'cnt_and_point', 'bounding', 'count_and_box', 'counting', 'pointing'}
import re

def get_d(s):
    try: return json.loads(re.search(r"```json\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE).group(1))
    except Exception: return []

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

def evaluate_samples(samples, task='cnt_and_point', thresh=10):
    if task not in ['counting','pointing','cnt_and_point','bounding','count_and_box']:
        raise ValueError("Invalid task")
    results = {}
    compute_iou = lambda A, B: (lambda xA,yA,xB,yB: (max(0,xB-xA+1)*max(0,yB-yA+1)) / ((A[2]-A[0]+1)*(A[3]-A[1]+1)+(B[2]-B[0]+1)*(B[3]-B[1]+1)-max(0,xB-xA+1)*max(0,yB-yA+1)) if ((A[2]-A[0]+1)*(A[3]-A[1]+1)+(B[2]-B[0]+1)*(B[3]-B[1]+1)-max(0,xB-xA+1)*max(0,yB-yA+1)) else 0)(max(A[0],B[0]), max(A[1],B[1]), min(A[2],B[2]), min(A[3],B[3]))
    
    if task in ['counting','cnt_and_point','count_and_box']:
        gt_counts = [s[0].get('counts',0) for s in samples]
        pred_counts = [s[1].get('counts',0) for s in samples]
        results['counting'] = {'count_mae': mean_absolute_error(gt_counts, pred_counts),
                                'count_mse': mean_squared_error(gt_counts, pred_counts)}
    if task in ['pointing','cnt_and_point']:
        pmae, prmse, macc, zc = [], [], [], 0
        for gt, pred in samples:
            g, p = np.array(gt.get('point_2d',[])), np.array(pred.get('point_2d',[]))
            if len(p)==0 or len(g[0])==0 or len(p[0])==0: zc+=1; continue
            d, _ = NearestNeighbors(n_neighbors=1).fit(g).kneighbors(p)
            pmae.append(d.mean()); prmse.append(np.sqrt((d**2).mean())); macc.append((d<thresh).sum()/len(p))
        results['pointing'] = {'point_mae': np.mean(pmae) if pmae else 0,
                               'point_rmse': np.mean(prmse) if prmse else 0,
                               'matching_accuracy': np.mean(macc) if macc else 0,
                               'zero_case_point': zc}
    if task in ['bounding','count_and_box']:
        targets = [{"boxes": torch.tensor(np.array(gt.get("bbox_2d", [])), dtype=torch.float), "labels": torch.ones(len(gt.get("bbox_2d", [])), dtype=torch.int64)} for gt, _ in samples]
        preds   = [{"boxes": torch.tensor(np.array(pred.get("bbox_2d", [])), dtype=torch.float), "scores": torch.ones(len(pred.get("bbox_2d", [])), dtype=torch.float), "labels": torch.ones(len(pred.get("bbox_2d", [])), dtype=torch.int64)} for _, pred in samples]
        map_metric = MeanAveragePrecision()
        map_metric.update(preds, targets)
        computed_map = map_metric.compute()
        ious = [box_iou(p["boxes"], t["boxes"]).max(dim=1)[0].mean() for p, t in zip(preds, targets) if p["boxes"].numel() and t["boxes"].numel()]
        results['bounding'] =  {
                    'mAP': computed_map['map'].item(),     # overall mAP
                    'mAP_50': computed_map['map_50'].item(), # mAP@50
                    'mAP_75': computed_map['map_75'].item(), # mAP@75
                    'iou': float(np.mean(ious)) if ious else 0.0  # average IoU (as float)
                }

    return results

from collections import defaultdict
evaluatees=defaultdict(list)
for idx, (gt, pred) in enumerate(zip(val_dataset, val_results)):
    # print(f"Index: {idx}")
    task, source, gt_labels = gt['messages'][0]['task'],gt['messages'][0]['source'], gt['messages'][1]['content']
    pred_labels = pred['response']
    gt_labels, pred_labels = get_d(gt_labels), get_d(pred_labels)
    # print(f"Task: {task}\n gt_labels {gt_labels} \n pred_labels {pred_labels}")

    # ent_pt    = mk_ent("pointing",    p_lbl, fmt_json([{"point_2d": [round(c, 1) for c in pt], "label": p_lbl} for pt in points]), "pointing")
    # ent_bb    = mk_ent("bounding",    b_lbl, fmt_json([{"bbox_2d": bb, "label": b_lbl} for bb in data['bbox']]), "bounding")
    # ent_cnt   = mk_ent("counting", c_lbl, fmt_json({"counts": data['counts'], "label": c_lbl}), "counting")
    # ent_cntpt = mk_ent("cnt_and_point", cp_lbl, fmt_json({"counts": data['counts'], "point_2d": [[round(c, 1) for c in pt]  for pt in points], "label": c_lbl, }), "cnt_and_point")
    # ent_cnt_bb = mk_ent("count_and_box", cp_lbl, fmt_json({"counts": data['counts'], "bbox_2d": [bb for bb in data['bbox']], "label": c_lbl, } ), "count_and_box")
   
    if task=='pointing':
       if type(pred_labels)==dict:
            pred_labels = [pred_labels]
       pred_labels = {"point_2d": [e.get('point_2d',[]) for e in pred_labels]}
       gt_labels = {"point_2d": [e.get('point_2d',[]) for e in gt_labels]}

    if task=='bounding':
       if type(pred_labels)==dict:
            pred_labels = [pred_labels]
       pred_labels = {"bbox_2d":[e.get('bbox_2d',[]) for e in pred_labels]}
       gt_labels = {"bbox_2d":[e.get('bbox_2d',[]) for e in gt_labels]}
    # if task=='counting':
    #     count = gt_labels.get('counts',0)
    if task=='cnt_and_point':
        if type(pred_labels)==list:
            if (pred_labels==[] or pred_labels==[[]]):
                pred_labels = {"counts": 0, "point_2d": []} 
            else:
                pred_labels = {"counts":len(pred_labels), "point_2d": [e.get('point_2d',[]) for e in pred_labels]}
        if pred_labels.get('point_2d')==[[]] or pred_labels.get('point_2d')==None:
            pred_labels['point_2d']=[]
    #     count = gt_labels.get('counts',0)
    #     point_2d = gt_labels.get('point_2d',[])
    if task=='count_and_box':
        if type(pred_labels)==list:
            if (pred_labels==[] or pred_labels==[[]]):
                pred_labels = {"counts": 0, "bbox_2d": []} 
            else:
                pred_labels = {"counts":len(pred_labels), "bbox_2d": [e.get('bbox_2d',[]) for e in pred_labels]}
        if pred_labels.get('bbox_2d')==[[]] or pred_labels.get('bbox_2d')==None:
            pred_labels['bbox_2d']=[]
    #     count = gt_labels.get('counts',0)
    #     bbox = gt_labels.get('bbox',[])
    if task=='counting':
        if type(pred_labels)==dict:
            pred_labels = {"counts": v for k, v in pred_labels.items() if 'count' in k and isinstance(v, int)}
        if type(pred_labels)==list and len(pred_labels) and (pred_labels[0].get('bbox_2d') or pred_labels[0].get('point_2d')):
            pred_labels={"counts":len(pred_labels)}
        if isinstance(pred_labels, list) and pred_labels and any("count" in key for key in pred_labels[0]):
            pred_labels={"counts": v for k, v in pred_labels[0].items() if 'count' in k and isinstance(v, int)}
        if pred_labels==[]: pred_labels={"counts":0}
        if type(pred_labels)==list:
            print(pred_labels)
            breakpoint()

    evaluatees[task].append((gt_labels, pred_labels))

evaluatees= dict(evaluatees)
for task, samples in sorted(evaluatees.items()):
    print(f"Task: {task}", len(samples))
    print(evaluate_samples(samples, task=task))
