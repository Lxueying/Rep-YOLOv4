#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import adabound

from torch.utils.data import DataLoader

from nets.ImprovedNeckYolo import YoloBody

# origin yolo
# from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                                weights_init)
from utils.callbacks import LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_anchors, get_classes
from utils.utils_fit import  fit_one_epoch

if __name__ == "__main__":
    Cuda = True
    print(torch.cuda.is_available())

    classes_path    = 'model_data/stra_classes.txt'

    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    model_path      = 'logs/FuseBranch.pth'

    input_shape     = [416, 416]
    # 预训练
    pretrained      = True

    mosaic              = True
    label_smoothing     = 0


    Init_Epoch          = 0
    Freeze_Epoch        = 20
    Freeze_batch_size   = 64

    UnFreeze_Epoch      = 20
    Unfreeze_batch_size = 32

    Freeze_Train        = True
    

    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01

    optimizer_type      = "sgd"
    momentum            = 0.937
    weight_decay        = 5e-4

    lr_decay_type       = "cos"

    save_period         = 1

    num_workers         = 4


    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'


    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)
    

    model = YoloBody(anchors_mask, num_classes, pretrained = pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':

        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model_dict      = model.state_dict()
        # model = torch.load(model_path, map_location = device)
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    yolo_loss    = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)
    loss_history = LossHistory("RepLogs/", model, input_shape=input_shape)
    
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()


    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)


    if True:
        UnFreeze_flag = False

        if Freeze_Train:
            # choose to freeze some parts of the net
            # for param in model.backbone.parameters():
            #     param.requires_grad = False
            for param in model.conv1.parameters():
                param.requires_grad = False
            for param in model.SPP.parameters():
                param.requires_grad = False
            for param in model.conv2.parameters():
                param.requires_grad = False
            for param in model.upsample1.parameters():
                param.requires_grad = False
            for param in model.conv_for_P4.parameters():
                param.requires_grad = False
            for param in model.make_five_conv1.parameters():
                param.requires_grad = False
            for param in model.upsample2.parameters():
                param.requires_grad = False
            for param in model.conv_for_P3.parameters():
                param.requires_grad = False
            for param in model.make_five_conv2.parameters():
                param.requires_grad = False
            for param in model.yolo_head3.parameters():
                param.requires_grad = False
            for param in model.down_sample1.parameters():
                param.requires_grad = False
            for param in model.make_five_conv3.parameters():
                param.requires_grad = False
            for param in model.yolo_head2.parameters():
                param.requires_grad = False
            for param in model.down_sample2.parameters():
                param.requires_grad = False
            for param in model.make_five_conv4.parameters():
                param.requires_grad = False
            for param in model.yolo_head1.parameters():
                param.requires_grad = False


        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


        nbs         = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)


        pg0, pg1, pg2 = [], [], []  
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)    
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)    
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)   
        optimizer = {
            'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
            'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True),
            'adabound':adabound.AdaBound(pg0, lr=0.001, final_lr= 0.1)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})


        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")


        train_dataset   = YoloDataset(train_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=mosaic, train = True)
        val_dataset     = YoloDataset(val_lines, input_shape, num_classes, epoch_length = UnFreeze_Epoch, mosaic=False, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate)


        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size


                nbs         = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                # for param in model.backbone.parameters():
                #     param.requires_grad = True
                for param in model.conv1.parameters():
                    param.requires_grad = True
                for param in model.SPP.parameters():
                    param.requires_grad = True
                for param in model.conv2.parameters():
                    param.requires_grad = True
                for param in model.upsample1.parameters():
                    param.requires_grad = True
                for param in model.conv_for_P4.parameters():
                    param.requires_grad = True
                for param in model.make_five_conv1.parameters():
                    param.requires_grad = True
                for param in model.upsample2.parameters():
                    param.requires_grad = True
                for param in model.conv_for_P3.parameters():
                    param.requires_grad = True
                for param in model.make_five_conv2.parameters():
                    param.requires_grad = True
                for param in model.yolo_head3.parameters():
                    param.requires_grad = True
                for param in model.down_sample1.parameters():
                    param.requires_grad = True
                for param in model.make_five_conv3.parameters():
                    param.requires_grad = True
                for param in model.yolo_head2.parameters():
                    param.requires_grad = True
                for param in model.down_sample2.parameters():
                    param.requires_grad = True
                for param in model.make_five_conv4.parameters():
                    param.requires_grad = True
                for param in model.yolo_head1.parameters():
                    param.requires_grad = True

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                gen     = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate)

                UnFreeze_flag = True

            gen.dataset.epoch_now       = epoch
            gen_val.dataset.epoch_now   = epoch

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, save_period)
