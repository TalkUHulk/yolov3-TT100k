# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from nets.yolo3 import YoloBody
from nets.yolo_training import YOLOLoss, LossHistory, weights_init
# from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.coco import COCO, yolo_dataset_collate, COCOEval
from utils.utils import DecodeBox, non_max_suppression
from utils.summary import Summary

yolo_decodes = []
confidence = 0.01
iou = 0.5
g_steps = 0


def get_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0

    net.train()
    print('Start Train')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break

            img_ids, images, targets = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    targets = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = net(images)
            losses = []
            num_pos_all = 0
            # ----------------------#
            #   计算损失
            # ----------------------#
            for i in range(3):
                loss_item, num_pos = yolo_loss(outputs[i], targets)
                losses.append(loss_item)
                num_pos_all += num_pos

            loss = sum(losses) / num_pos_all
            # ----------------------#
            #   反向传播
            # ----------------------#
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            summary.add_scalar('train/total_loss', total_loss / (iteration + 1), epoch * epoch_size + iteration)
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Train')

    net.eval()
    print('Start Validation')
    img_ids = []
    detections = []
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            img_ids_val, images_val, targets_val = batch[0], batch[1], batch[2]
            with torch.no_grad():
                if cuda:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor).cuda()
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                else:
                    images_val = torch.from_numpy(images_val).type(torch.FloatTensor)
                    targets_val = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in targets_val]
                optimizer.zero_grad()

                outputs = net(images_val)

                # ----------------------#
                #   计算mAP
                # ----------------------#
                output_list = []
                for ii in range(3):
                    output_list.append(yolo_decodes[ii](outputs[ii]))

                output = torch.cat(output_list, 1)
                batch_detections = non_max_suppression(output, num_classes, conf_thres=confidence,
                                                       nms_thres=iou)
                # -----------------------------------debug---------------------------------
                # import cv2
                # images = images_val.squeeze().cpu().numpy()
                # images = np.transpose(np.clip(images * 255.0, 0, 255), (1, 2, 0)).astype(np.uint8)
                # cv2.imshow("images", images)
                # cv2.waitKey()
                # print(img_ids_val, batch_detections)
                # -----------------------------------debug---------------------------------
                img_ids += img_ids_val
                detections += batch_detections
                losses = []
                num_pos_all = 0

                # ----------------------#
                #   计算损失
                # ----------------------#
                for i in range(3):
                    loss_item, num_pos = yolo_loss(outputs[i], targets_val)
                    losses.append(loss_item)
                    num_pos_all += num_pos

                loss = sum(losses) / num_pos_all
                val_loss += loss.item()

            summary.add_scalar('val/total_loss', val_loss / (iteration + 1), epoch * epoch_size + iteration)
            pbar.set_postfix(**{'total_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    eval_results = val_dataset.run_eval(img_ids, detections)
    loss_history.append_loss(total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1))
    summary.add_scalar('val/mAP', eval_results[0], epoch + 1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print(eval_results)
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))
    if (epoch + 1) % 10 == 0:
        print('Saving state, iter:', str(epoch + 1))
        torch.save(model.state_dict(), '%s/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
            os.path.join(exp_dir, "ckpt"), (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


# ----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
# ----------------------------------------------------#
if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ------------------------------------------------------#
    #   是否对损失进行归一化，用于改变loss的大小
    #   用于决定计算最终loss是除上batch_size还是除上正样本数量
    # ------------------------------------------------------#
    normalize = False
    # ------------------------------------------------------#
    #   输入的shape大小
    # ------------------------------------------------------#
    # input_shape = (416, 416)
    input_shape = (608, 608)
    # ------------------------------------------------------#
    #   视频中的Config.py已经移除
    #   需要修改num_classes直接修改此处的num_classes即可
    #   如果需要检测5个类, 这里就写5. 默认为20
    # ------------------------------------------------------#
    num_classes = 151
    # ----------------------------------------------------#
    #   先验框anchor的路径
    # ----------------------------------------------------#
    anchors_path = 'model_data/tt100k_anchors.txt'
    anchors = get_anchors(anchors_path)

    # ----------------------------------------------------#

    for i in range(3):
        yolo_decodes.append(
            DecodeBox(anchors[i], num_classes, (input_shape[1], input_shape[0])))

    # ------------------------------------------------------#
    #   创建yolo模型
    #   训练前一定要修改Config里面的classes参数
    # ------------------------------------------------------#
    model = YoloBody(anchors, num_classes)
    weights_init(model)

    # ------------------------------------------------------#
    #   权值文件请看README，百度网盘下载
    # ------------------------------------------------------#
    model_path = "model_data/yolo_weights.pth"
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    yolo_loss = YOLOLoss(np.reshape(anchors, [-1, 2]), num_classes, (input_shape[1], input_shape[0]), Cuda, normalize)
    exp_dir = "./exp/ft_608_1"
    loss_history = LossHistory(exp_dir)
    summary = Summary(os.path.join(exp_dir, "summary"))
    ckpt = os.path.join(exp_dir, "ckpt")
    if not os.path.exists(ckpt):
        os.makedirs(ckpt)

    # ----------------------------------------------------#
    #   获得图片路径和标签
    # ----------------------------------------------------#
    data_dir = "/Users/hulk/Documents/DataSets/TT100k"
    # data_dir = "../tt100k/"

    # ------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Unfreeze_Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 32
        Init_Epoch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = COCO(data_dir, (input_shape[0], input_shape[1]))
        # val_dataset = COCO(data_dir, (input_shape[0], input_shape[1]), False)
        val_dataset = COCOEval(data_dir, (input_shape[0], input_shape[1]))
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        epoch_size = len(train_dataset) // Batch_size
        epoch_size_val = len(val_dataset) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 16
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        train_dataset = COCO(data_dir, (input_shape[0], input_shape[1]))
        val_dataset = COCOEval(data_dir, (input_shape[0], input_shape[1]))
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                         drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val = DataLoader(val_dataset, shuffle=True, batch_size=Batch_size, num_workers=4, pin_memory=False,
                             drop_last=True, collate_fn=yolo_dataset_collate)

        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        epoch_size = len(train_dataset) // Batch_size
        epoch_size_val = len(val_dataset) // Batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()


