import argparse
from audioop import mul
import json
import os,cv2
from re import T
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, combine_box, get_roi_res, get_roi_res_torch, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, xywh2xyxy1d, set_logging, increment_path, colorstr, remove_inside_box
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.road_metrics import road_ap_per_class
from utils.plots import plot_images, output_to_target, plot_study_txt, colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

@torch.no_grad()
def test(data,
         weights=None,  # model.pt path(s)
         batch_size=32,  # batch size
         imgsz=640,  # inference size (pixels)
         conf_thres=0.001,  # confidence threshold
         iou_thres=0.6,  # NMS IoU threshold
         task='val',  # train, val, test, speed or study
         device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
         single_cls=False,  # treat as single-class dataset
         augment=False,  # augmented inference
         verbose=False,  # verbose output
         save_txt=False,  # save results to *.txt
         save_hybrid=False,  # save label+prediction hybrid results to *.txt
         save_img=False,  #save img in --save-img
         save_conf=False,  # save confidences in --save-txt labels
         save_json=False,  # save a cocoapi-compatible JSON results file
         project='runs/test',  # save to project/name
         name='exp',  # save to project/name
         exist_ok=False,  # existing project/name ok, do not increment
         half=False,  # use FP16 half-precision inference
         model=None,
         dataloader=None,
         save_dir=Path(''),
         plots=True,
         wandb_logger=None,
         compute_loss=None,
         roi=False
         ):
    show_classes = [] # class appeared in test data
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        set_logging()
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    if isinstance(data, str):
        with open(data) as f:
            data = yaml.safe_load(f)
    check_dataset(data)  # check
    is_coco = data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Logging
    log_imgs = 0
    if wandb_logger and wandb_logger.wandb:
        log_imgs = min(wandb_logger.log_imgs, 100)
    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    if len(names)==7:
        print('-------------------------------' + 'Liqing mode' + '---------------------------------')
        modedir = "runs/detect/" + "liqing" + "/"
        multi_confs = { 0 : 0.2, 1 : 0.05, 2 : 0.05, 3 : 0.2, 4 : 0.1, 5 : 0.2, 6 : 0.2 }
        print(multi_confs)
        # multi_confs = { 0 : 0.05, 1 : 0.05, 2 : 0.05, 3 : 0.05, 4 : 0.05, 5 : 0.05, 6 : 0.05 }
    else:
        print('-------------------------------' + 'shuini mode' + '---------------------------------')
        modedir = "runs/detect/" + "shuini" + "/"
        multi_confs = { 0 : 0.1, 1 : 0.025, 2 : 0.05, 3 : 0.05, 4 : 0.05, 5 : 1.0, 6 : 0.05, 7 : 0.05 }
    if not os.path.exists(modedir):
        os.makedirs(modedir)

    tp_nums_classes, pp_nums_classes, gt_nums_total, pred_nums_total, prelist, recalllist = {}, {}, {}, {}, {}, {}
    # insert by zhinanzhang
    for i in range(len(names)):
        tp_nums_classes[i] = 0
        pp_nums_classes[i] = 0
        gt_nums_total[i] = 0
        pred_nums_total[i] = 0
        prelist[i] = []
        recalllist[i] = []

    # coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%11s' * 7) % ('Class', 'Labels', 'Images', 'Thres', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class, wandb_images = [], [], [], [], []
    
    for batch_i, (img, img0s, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_synchronized()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_synchronized()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_synchronized() - t
        
        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_synchronized()
        if training:
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls, multi_confs={})
        else:
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=False, agnostic=single_cls, multi_confs=multi_confs)
        t2 += time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(out):
            img0 = img0s[si]
            img1 = img0.copy()
            height = img1.shape[0]
            img2 = img0.copy()
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1
            # insert by zhinanzhang 0114
            predn = pred.clone()
            if nl:
                scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

            if roi:
                labels = get_roi_res_torch(labels, height, 3/4, xyxy=False)
                pred = get_roi_res_torch(pred, height, 3/4) # test input shape (640, 640, 3)

            draw = pred.clone()
            if len(pred) != 0:
                # Predictions
                draw = combine_box(draw, 0.3, classes=[0, 1, 2, 3, 4, 5, 6])
                draw = remove_inside_box(draw, classes=[0, 1, 2, 3, 4, 5, 6])
            
            # Assign all predictions as incorrect
            if nl:
                labeln = labels.clone()
                tcls_tensor = labels[:, 0]
                # postprocess edited by zhinanzhang 
                if save_img:
                    index = 0
                    for cls, *xywh in labeln.tolist():
                        tbox = xywh2xyxy1d(xywh)
                        c = int(cls)  # integer class
                        label = f'{names[c]}'
                        plot_one_box(tbox, img0, label=label, color=colors(c, True), line_thickness=1)
                    
                    for *xyxy, conf, cls in draw: # pred.tolist() combine_predn
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, img1, label=label, color=colors(c, True), line_thickness=1)
                        index += 1
                    img3 = np.hstack([img1, img0])  #left pred;right label
                    save_name = modedir + path.stem +".jpg"
                    cv2.imwrite(save_name, img3)

                    # for *xyxy, conf, cls in pred.tolist(): # pred.tolist() combine_predn
                    #     c = int(cls)  # integer class
                    #     label = f'{names[c]} {conf:.2f}'
                    #     plot_one_box(xyxy, img2, label=label, color=colors(c, True), line_thickness=1)
                    # img4 = np.hstack([img2, img0])  #left pred;right label
                    # save_name = modedir + path.stem + "_true.jpg"
                    # cv2.imwrite(save_name, img4)

                pcls_list = pred[:, 5].cpu().numpy()
                # pcls_list = combine_predn[:, 5]
                tcls_list = tcls_tensor.cpu().numpy()

                pcls_list, pcount = np.unique(pcls_list,return_counts=True)
                tcls_list, tcount = np.unique(tcls_list,return_counts=True)
                tdict = dict(zip(tcls_list,tcount))
                pdict = dict(zip(pcls_list,pcount))
                for tcls in tdict:
                    newtcls = tcls
                    if newtcls not in show_classes:
                        show_classes.append(newtcls)
                    gt_nums_total[newtcls] += 1 #TP+FN
                    if tcls in pdict:
                        tp_nums_classes[newtcls] += 1 #TP
                for pcls in pdict:
                    newpcls = pcls
                    pred_nums_total[newpcls] += pdict[pcls]
                    if pcls in tdict:
                        pp_nums_classes[newpcls] += pdict[pcls]
                for key in prelist:
                    prelist[key].append(round(pp_nums_classes[key] / (pred_nums_total[key] + 1e-15), 4))
                    recalllist[key].append(round(tp_nums_classes[key] / (gt_nums_total[key] + 1e-15), 4))
                
        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'test_batch{batch_i}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    p, r, ap, ap_class = road_ap_per_class(recalllist, prelist, batch_i+1)
    ap50, ap = ap.mean(1), ap.mean(1)  # AP@0.5, AP@0.5:0.95
    newp, newr, newap, newap50 = [], [], [], []
    for i, c in enumerate(ap_class):
        if c in show_classes:
            newp.append(p[i])
            newr.append(r[i])
            newap.append(ap[i])
            newap50.append(ap[i])
    newp, newr, newap, newap50 = np.array(newp), np.array(newr), np.array(newap), np.array(newap50)
    mp, mr, map50, map = newp.mean(), newr.mean(), newap50.mean(), newap.mean()

    nt = []
    for k in gt_nums_total:
        nt.append(gt_nums_total[k])
    nt = np.array(nt)    

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 5  # print format
    print(pf % ('all', seen, nt.sum(), conf_thres, mp, mr, map50, map))
    
    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(nt):
        for i, c in enumerate(ap_class):
            if c in show_classes:
                if not training:
                    print(pf % (names[c], seen, nt[c], multi_confs[c], p[i], r[i], ap50[i], ap[i]))
                else:
                    print(pf % (names[c], seen, nt[c], conf_thres, p[i], r[i], ap50[i], ap[i]))

    # insert by zhinanzhang
    # print total pre recall
    if not training:
        f = '%20s' + '%11g' * 2 # print format
        sf = '%20s' + '%11s' * 2
        print("--"*50 + "\n")
        print(sf % ("Class", "P", "R"))
        for c in show_classes:
            print(f % (names[c], prelist[c][-1],recalllist[c][-1]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        if wandb_logger and wandb_logger.wandb:
            val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(save_dir.glob('test*.jpg'))]
            wandb_logger.log({"Validation": val_batches})
    if wandb_images:
        wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = '../coco/annotations/instances_val2017.json'  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print('\nEvaluating pycocotools mAP... saving %s...' % pred_json)
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', default=True, help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-img', action='store_true', help='save img to detect')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--roi', action='store_true', help='use roi to postprocess preds and labels')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        test(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            test(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                 save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python test.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = test(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                               iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot
