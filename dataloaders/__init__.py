# from dataloaders.datasets import cityscapes, kd, coco, combine_dbs, pascal, sbd
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from SimpleITK import *
import SimpleITK as sitk
from os.path import join
import numpy as np
import math
def make_data_loader(args, **kwargs):
    data_dict={}
    num_class = 4
    data_dir = './data'
    # out_base = join(nnUNet_raw_data, foldername)
    # imagestr = join(out_base, "imagesTr")
    # imagests = join(out_base, "imagesTs")
    # labelstr = join(out_base, "labelsTr")

    training_data = sitk.GetArrayFromImage(sitk.ReadImage(join(data_dir, 'tr_im.nii.gz')))
    training_labels = sitk.GetArrayFromImage(sitk.ReadImage(join(data_dir, 'tr_mask.nii.gz')))
    # test_data = sitk.GetArrayFromImage(sitk.ReadImage(join(data_dir, 'val_im.nii.gz')))
    # test_labels = sitk.GetArrayFromImage(sitk.ReadImage(join(data_dir, 'tr_mask.nii.gz')))

    train_data = []
    train_mask = []

    valid_data = []
    valid_mask = []
    for f in range(100):
        this_name = 'part_%d' % f
        data = training_data[f, :, :]
        data = np.reshape(data, [1, 1, data.shape[0], data.shape[1]])

        tmp_labels = training_labels[f, :, :]
        tmp_labels = np.array(tmp_labels)
        labels = np.reshape(tmp_labels, [1, tmp_labels.shape[0], tmp_labels.shape[1]])

        # labels = np.zeros([1, 4, tmp_labels.shape[0], tmp_labels.shape[1]])
        # for i in range(num_class):
        #     class_i = np.where(tmp_labels == i)
        #     labels[:, i, class_i[0], class_i[1]] = 1

        # sitk.WriteImage(sitk.GetImageFromArray(data), this_name + 'tr_im.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(labels), join(labelstr, this_name + '.nii.gz'))
        if f >= 80:
            valid_data.append(data)
            valid_mask.append(labels)
        else:
            train_data.append(data)
            train_mask.append(labels)

    data__dict={}
    data_dict['train_data'] = train_data
    data_dict['train_mask'] = train_mask
    data_dict['valid_data'] = valid_data
    data_dict['valid_mask'] = valid_mask
    data_dict['num_train'] = 80
    data_dict['num_valid'] = 20
    return data_dict, num_class
def make_data_loader_3d_patch(args, **kwargs):
   
    data_dict = {}
    num_class = 4
    data_dir = './data/COVID-19-CT-Seg_20cases'
    mask_dir = './data/Lung_and_Infection_Mask'
    # out_base = join(nnUNet_raw_data, foldername)
    # imagestr = join(out_base, "imagesTr")
    # imagests = join(out_base, "imagesTs")
    # labelstr = join(out_base, "labelsTr")

    train_data = []
    train_mask = []

    valid_data = []
    valid_mask = []
    test_data = []

    for i in range(1, 11):
        training_data = sitk.GetArrayFromImage(
            sitk.ReadImage(join(data_dir, 'coronacases_0' + str(i).zfill(2) + '.nii.gz')))
        training_labels = sitk.GetArrayFromImage(
            sitk.ReadImage(join(mask_dir, 'coronacases_0' + str(i).zfill(2) + '.nii.gz')))

        patch_size = (5, 128, 128)

        index0 = get_split_index(patch_size[0], training_data.shape[0], drop_last=True)
        index1 = get_split_index(patch_size[1], training_data.shape[1], drop_last=True)
        index2 = get_split_index(patch_size[2], training_data.shape[2], drop_last=True)

        data = []
        labels = []
        for i0 in index0:
            for i1 in index1:
                for i2 in index2:
                    data_tmp = training_data[i0:i0 + patch_size[0], i1:i1 + patch_size[1], i2:i2 + patch_size[2]]
                    label_tmp = training_labels[i0:i0 + patch_size[0], i1:i1 + patch_size[1], i2:i2 + patch_size[2]]
                    data.append(np.reshape(data_tmp, [1, 1, data_tmp.shape[0], data_tmp.shape[1], data_tmp.shape[2]]))
                    labels.append(
                        np.reshape(label_tmp, [1, label_tmp.shape[0], label_tmp.shape[1], label_tmp.shape[2]]))


        if i > 8:
            valid_data += data
            valid_mask += labels
        else:
            train_data += data
            train_mask += labels

    data__dict = {}
    data_dict['train_data'] = train_data
    data_dict['train_mask'] = train_mask
    data_dict['valid_data'] = valid_data
    data_dict['valid_mask'] = valid_mask
    data_dict['num_train'] = 8
    data_dict['num_valid'] = 2
    return data_dict, num_class


# overlap: set overlap length
# drop_last: whether to drop the last incomplete patch, set True then the last patch will overlap backwards

def get_split_index(split_size, total_size, overlap=0, drop_last=False):
    num = math.ceil((total_size - split_size) / (split_size - overlap))
    if (total_size - split_size) / (split_size - overlap) == int((total_size - split_size) / (split_size - overlap)):
        num += 1
    overlap_size = overlap

    index = [0]
    for i in range(int(num - 1)):
        index.append(int(index[i] + split_size - overlap_size))

    if drop_last:
        return index
    else:
        if (total_size - split_size) / (split_size - overlap) > int((total_size - split_size) / (split_size - overlap)):
            index.append(int(total_size - split_size))
        return index
    return
def make_data_loader_seg(args, **kwargs):
    if args.dist:
        print("=> Using Distribued Sampler")
        if args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set1)
                sampler2 = torch.utils.data.distributed.DistributedSampler(train_set2)
                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)
                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=False, sampler=sampler2, **kwargs)

            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
                num_class = train_set.NUM_CLASSES
                sampler1 = torch.utils.data.distributed.DistributedSampler(train_set)
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=sampler1, **kwargs)

            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            sampler3 = torch.utils.data.distributed.DistributedSampler(val_set)
            sampler4 = torch.utils.data.distributed.DistributedSampler(test_set)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, sampler=sampler3, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, sampler=sampler4, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader, num_class, sampler1
        else:
            raise NotImplementedError

    else:
        if args.dataset == 'pascal':
            train_set = pascal.VOCSegmentation(args, split='train')
            val_set = pascal.VOCSegmentation(args, split='val')
            if args.use_sbd:
                sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
                train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None

            return train_loader, train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'cityscapes':
            if args.autodeeplab == 'search':
                train_set1, train_set2 = cityscapes.twoTrainSeg(args)
                num_class = train_set1.NUM_CLASSES
                train_loader1 = DataLoader(train_set1, batch_size=args.batch_size, shuffle=True, **kwargs)
                train_loader2 = DataLoader(train_set2, batch_size=args.batch_size, shuffle=True, **kwargs)
            elif args.autodeeplab == 'train':
                train_set = cityscapes.CityscapesSegmentation(args, split='retrain')
                num_class = train_set.NUM_CLASSES
                train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            else:
                raise Exception('autodeeplab param not set properly')

            val_set = cityscapes.CityscapesSegmentation(args, split='val')
            test_set = cityscapes.CityscapesSegmentation(args, split='test')
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            if args.autodeeplab == 'search':
                return train_loader1, train_loader2, val_loader, test_loader, num_class
            elif args.autodeeplab == 'train':
                return train_loader, num_class



        elif args.dataset == 'coco':
            train_set = coco.COCOSegmentation(args, split='train')
            val_set = coco.COCOSegmentation(args, split='val')
            num_class = train_set.NUM_CLASSES
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = None
            return train_loader, train_loader, val_loader, test_loader, num_class

        elif args.dataset == 'kd':
            train_set = kd.CityscapesSegmentation(args, split='train')
            val_set = kd.CityscapesSegmentation(args, split='val')
            test_set = kd.CityscapesSegmentation(args, split='test')
            num_class = train_set.NUM_CLASSES
            train_loader1 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            train_loader2 = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

            return train_loader1, train_loader2, val_loader, test_loader, num_class
        else:
            raise NotImplementedError


