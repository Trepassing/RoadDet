import argparse
from audioop import mul
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms as T

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, get_roi_res_torch, get_roi_res_seg, \
    scale_coords, segment2box, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box, combine_box, remove_inside_box, scale_coords_2d
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

import network
from network import deeplabv3_mobilenet

def decode_target(target):
    train_id_to_color = [[128, 64, 128], [153, 153, 153], [0, 0, 0]]
    train_id_to_color = np.array(train_id_to_color)
    target[target == 255] = 2
    decoded = train_id_to_color[target]
    decoded = decoded.astype(np.uint8)

    decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    g = cv2.getStructuringElement(cv2.MORPH_RECT, (11,11))
    mask = cv2.morphologyEx(decoded, cv2.MORPH_OPEN, g)
    _, binary = cv2.threshold(mask,130,255,cv2.THRESH_BINARY)
    return binary

@torch.no_grad()
# insert by zhinanzhang  add roi_thres 20220119
def detect(roi_thres=0.75,
           sweights='checkpoints/best_deeplabv3plus_mobilenet_roadDet_os16.pth', # Seg model path
           weights='yolov5s.pt',  # Det model.pt path(s)
           source='data/images',  # file/dir/URL/glob, 0 for webcam
           imgsz=640,  # inference size (pixels)
           conf_thres=0.25,  # confidence threshold
           iou_thres=0.45,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
           view_img=False,  # show results
           save_txt=False,  # save results to *.txt
           save_conf=False,  # save confidences in --save-txt labels
           save_crop=False,  # save cropped prediction boxes
           nosave=False,  # do not save images/videos
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           update=False,  # update all models
           project='runs/detect',  # save results to project/name
           name='exp',  # save results to project/name
           exist_ok=False,  # existing project/name ok, do not increment
           line_thickness=3,  # bounding box thickness (pixels)
           hide_labels=False,  # hide labels
           hide_conf=False,  # hide confidences
           half=True,  # use FP16 half-precision inference
           ):
    save_img = not nosave and not source.endswith('.txt')  # save inference images

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model Det + Seg
    detModel = attempt_load(weights, map_location=device)  # load FP32 model
    segModel = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=2, output_stride=16)
    # network.convert_to_separable_conv(segModel.classifier)
    checkpoint = torch.load(sweights, map_location=device)
    segModel.load_state_dict(checkpoint["model_state"])
    segModel.to(device)
    segModel.eval()

    stride = int(detModel.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    names = detModel.module.names if hasattr(detModel, 'module') else detModel.names  # get class names

    # insert by zhinanzhang 20220119
    if len(names) == 7:
        print('-' * 25 + 'Liqing mode' + '-' * 25)
        multi_confs = { 0 : 0.5, 1 : 0.5, 2 : 0.35, 3 : 0.9, 4 : 0.5, 5 : 0.2, 6 : 0.3 }
        # multi_confs = { 0 : 0.2, 1 : 0.1, 2 : 0.05, 3 : 0.4, 4 : 0.1, 5 : 0.2, 6 : 0.3 }
    else:
        print("MODEL ERROR")
        assert False
    if half:
        detModel.half()  # to FP16
        segModel.half()
    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        segModel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(detModel.parameters())))  # run once
        detModel(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(detModel.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # insert by zhinanzhang height applyed in line127 20220119
        height = im0s.shape[0]
        img = img.astype('float')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        # seg pipeline
        # transform = T.Compose([
        #         T.ToTensor(),
        #         T.Normalize(mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225]),
        #     ])
        # img = transform(img).unsqueeze(0) # To tensor of NCHW

        simg = (img - [[[0.485]], [[0.456]], [[0.406]]]) / [[[0.229]], [[0.224]], [[0.225]]]
        simg = torch.from_numpy(simg).to(device)
        simg = simg.half() if half else simg.float()  # uint8 to fp16/32
        if simg.ndimension() == 3:
            simg = simg.unsqueeze(0)

        segPred = segModel(simg)
        segPred = segPred.max(1)[1].cpu().numpy()[0] # HW

        colorized_preds = decode_target(segPred).astype('uint8')
        contours, _ = cv2.findContours(colorized_preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # get region of interest by segModel  ROI:contour
        RoI, maxArea = np.array([]), 0
        for contour in contours:
            area = cv2.contourArea(contour)
            # 可以加入是否大于一定面积的判断
            if area > maxArea:
                RoI = contour
                maxArea = area

        if len(RoI) != 0:
            RoI = scale_coords_2d(simg.shape[2:], RoI.astype('float'), im0s.shape).astype('int')
        # det pipeline
        dimg = torch.from_numpy(img).to(device)
        dimg = dimg.half() if half else dimg.float()  # uint8 to fp16/32
        if dimg.ndimension() == 3:
            dimg = dimg.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = detModel(dimg, augment=augment)[0]
        
        # Apply NMS
        # changed by zhinanzhang line101
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, multi_confs=multi_confs)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % dimg.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop

            zeros = np.zeros((im0.shape), dtype=np.uint8)
            cv2.drawContours(zeros, [RoI], -1, (100, 0, 255), -1)
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(dimg.shape[2:], det[:, :4], im0.shape).round()
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                detn = det.clone()
                # insert by zhinanzhang line128 20220119
                if len(RoI) != 0:
                    detn = get_roi_res_seg(detn, RoI, list(multi_confs.keys())[-2])
                else:
                    detn = get_roi_res_torch(detn, height, roi_thres, list(multi_confs.keys())[-2])
                # insert by zhinanzhang line130 20220112
                combine_predn = remove_inside_box(detn, classes=[0, 1, 2, 3, 4, 5, 6])

                # Write results
                for *xyxy, conf, cls in reversed(combine_predn):  # det combine_predn
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    im0 = 0.3 * zeros + im0
                    im0 = cv2.resize(im0, None, fx=0.5, fy=0.5)
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # insert by zhinanzhang add roi-thres
    parser.add_argument('--roi-thres', type=float, default=0.75, help='the roi ratio of height')
    parser.add_argument('--sweights', nargs='+', type=str, default='checkpoints/best_deeplabv3plus_mobilenet_roadDet_os16.pth', help='seg model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='checkpoints/yolo_best.pt', help='det model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.01, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'thop'))

    detect(**vars(opt))
