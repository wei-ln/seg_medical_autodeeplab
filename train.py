import os
import pdb
import warnings
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
from dataloaders import make_data_loader, make_data_loader_3d_patch

from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args

from utils.metrics import Evaluator

def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    args.data_dict={}
    model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
    if args.dataset == 'pascal':
        raise NotImplementedError
    elif args.dataset == 'cityscapes':
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
        dataset_loader, num_classes = make_data_loader(args, **kwargs)
        args.num_classes = num_classes
    elif args.dataset =='2d':
        args.data_dict, args.num_classes = make_data_loader(args)
    elif args.dataset =='3d':
        args.data_dict, args.num_classes = make_data_loader_3d_patch(args)

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.criterion == 'Ohem':
        args.thresh = 0.7
        args.crop_size = [args.crop_size, args.crop_size] if isinstance(args.crop_size, int) else args.crop_size
        args.n_min = int((args.batch_size / len(args.gpu) * args.crop_size[0] * args.crop_size[1]) // 16)
    criterion = build_criterion(args)

    model = nn.DataParallel(model).cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.module.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = args.data_dict['num_train'] * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, args.data_dict['num_train'])
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        for i in range(args.data_dict['num_train']):
            cur_iter = epoch * args.data_dict['num_train'] + i
            scheduler(optimizer, cur_iter)
            inputs = torch.FloatTensor(args.data_dict['train_data'][i]).cuda()
            target = torch.FloatTensor(args.data_dict['train_mask'][i]).cuda()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                pdb.set_trace()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, i + 1, args.data_dict['num_train'], scheduler.get_lr(optimizer), loss=losses))

        if epoch < args.epochs - 50:
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, model_fname % (epoch + 1))
        else:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, model_fname % (epoch + 1))

        if epoch % 2 ==0:
            ########################valid and test
            validation(epoch, model, args, criterion, args.num_classes)
            validation(epoch, model, args, criterion, args.num_classes, test_tag=True)
        print('reset local total loss!')


def validation(epoch,model,args,criterion,nclass,test_tag=False):
    model.eval()

    losses = 0.0

    evaluator = Evaluator(nclass)
    evaluator.reset()
    if test_tag==True:
        num_img = args.data_dict['num_valid']
    else:
        num_img = args.data_dict['num_test']
    for i in range(num_img):
        if test_tag==True:
            inputs = torch.FloatTensor(args.data_dict['valid_data'][i]).cuda()
            target = torch.FloatTensor(args.data_dict['valid_mask'][i]).cuda()
        else:
            inputs = torch.FloatTensor(args.data_dict['test_data'][i]).cuda()
            target = torch.FloatTensor(args.data_dict['test_mask'][i]).cuda()

        with torch.no_grad():
            output = model(inputs)
        loss_val = criterion(output, target)
        print(
            'epoch: {0}\t''iter: {1}/{2}\t''loss: {loss:.4f}'.format(epoch + 1, i + 1, args.data_dict['num_train'], loss= loss_val))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)
 
        losses += loss_val

        if test_tag == True:
            #save input,target,pred
            pred_save_dir = './pred/'
            sitk.WriteImage(sitk.GetImageFromArray(inputs), pred_save_dir + 'input_{}.nii.gz'.format(i))
            sitk.WriteImage(sitk.GetImageFromArray(target), pred_save_dir + 'target_{}.nii.gz'.format(i))
            sitk.WriteImage(sitk.GetImageFromArray(pred), pred_save_dir + 'pred_{}_{}.nii.gz'.format(i, epoch))

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
    if test_tag==True:
        print('Test:')
    else:
        print('Validation:')
    print('[Epoch: %d, numImages: %5d]' % (epoch, num_img))
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % losses)

    # val_losses+=loss_val
    # print('epoch: {0}\t''loss: {loss.val:.4f}'.format(epoch + 1, val_losses))


if __name__ == "__main__":
    main()



