import os
import time
import numpy as np
from argparse import ArgumentParser
import torch
from torch.nn.utils import clip_grad_norm_
from utils.sc_metric_tool import SCConfuseMatrixMeter
from utils.bs_metric_tool import BSConfuseMatrixMeter
import datasets.dataset_GF as myDataLoader1
import datasets.dataset_SECOND as myDataLoader
from models.initial_model import get_model_by_name, adjust_learning_rate
from utils.loss import ChangeSimilarity, DeepSupervisionLoss, MutilCrossEntropyDiceLoss, MutilCrossEntropyLoss


@torch.no_grad()
def validate(args, val_loader, model, criterion_sc, criterion_bc, criterion_con):
    model.eval()
    sc_evaluation = SCConfuseMatrixMeter(n_class=args.num_classes)
    bc_evaluation = BSConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    total_batches = len(val_loader)
    print(f'Total batches for validation: {total_batches}')

    for iter, batched_inputs in enumerate(val_loader):
        start_time = time.time()

        pre_img, post_img, pre_target, post_target,  img_names = batched_inputs
        pre_img, post_img, pre_target, post_target = map(lambda x: x.cuda(),
                                                         [pre_img, post_img, pre_target, post_target])

        binary_target = (pre_target != post_target).float()
        binary_mask = (pre_target != post_target).float()
        labeled_mask=(pre_target >= 0).float()

        # run the model
        pre_mask, post_mask, change_mask = model(pre_img, post_img)

        # loss
        loss_seg = criterion_sc(pre_mask, pre_target) + criterion_sc(post_mask, post_target)
        loss_bn = criterion_bc(change_mask, binary_mask)
        loss_sc = criterion_con(pre_mask, post_mask, binary_mask)
        loss = loss_seg * 0.5 + loss_bn + loss_sc

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # Computing Performance
        with torch.no_grad():
            change_mask = change_mask[:, 0:1]
            change_mask = torch.sigmoid(change_mask)
            change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask),
                                      torch.zeros_like(change_mask)).long()
            pre_mask = torch.argmax(pre_mask, dim=1)
            post_mask = torch.argmax(post_mask, dim=1)
            mask = torch.cat(
                [pre_mask * change_mask.squeeze().long(),
                 post_mask * change_mask.squeeze().long()], dim=0
            )
            pre_target = pre_target * binary_target
            post_target = post_target*binary_target
            mask_gt = torch.cat([pre_target, post_target], dim=0)
            o_score = sc_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())
            f1 = bc_evaluation.update_cm(pr=change_mask.cpu().numpy(), gt=binary_mask.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] score: %2f loss: %.3f time: %.3f' %
                  (iter, total_batches, o_score, loss.data.item(), time_taken), end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    sc_scores = sc_evaluation.get_scores()
    bc_scores = bc_evaluation.get_scores()

    return average_epoch_loss_val, sc_scores, bc_scores


def train(args, train_loader, model, criterion_sc, criterion_bc, criterion_con, scaler, optimizer, max_batches,
          cur_iter=0):
    model.train()
    sc_evaluation = SCConfuseMatrixMeter(n_class=args.num_classes)
    bc_evaluation = BSConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):
        start_time = time.time()

        pre_img, post_img, pre_target, post_target, img_names = batched_inputs
        pre_img, post_img, pre_target, post_target = map(lambda x: x.cuda(),
                                                         [pre_img, post_img, pre_target, post_target])
        binary_target = (pre_target != post_target).float()
        binary_mask = (pre_target != post_target).float()
        labeled_mask = (pre_target >= 0).float()

        lr = adjust_learning_rate(args, optimizer, iter + cur_iter, max_batches)
        with torch.cuda.amp.autocast():
            # run the model
            pre_mask, post_mask, change_mask = model(pre_img, post_img)

            # loss
            pre_mask, post_mask, change_mask = map(lambda x: x.float(), [pre_mask, post_mask, change_mask])
            loss_seg = criterion_sc(pre_mask, pre_target) + criterion_sc(post_mask,post_target)
            loss_bn = criterion_bc(change_mask, binary_mask)
            loss_sc = criterion_con(pre_mask, post_mask, binary_mask)
            loss = loss_seg * 0.5 + loss_bn + loss_sc 

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        # Computing Performance
        with torch.no_grad():
            change_mask = change_mask[:, 0:1]
            change_mask = torch.sigmoid(change_mask)
            change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask),
                                      torch.zeros_like(change_mask)).long()

            pre_target = pre_target*binary_target
            post_target = post_target*binary_target

            pre_mask = torch.argmax(pre_mask, dim=1)
            post_mask = torch.argmax(post_mask, dim=1)
            mask = torch.cat(
                [pre_mask * change_mask.squeeze().long(),
                 post_mask * change_mask.squeeze().long()], dim=0
            )
            mask_gt = torch.cat([pre_target, post_target], dim=0)
            o_score = sc_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())
            f1 = bc_evaluation.update_cm(pr=change_mask.cpu().numpy(), gt=binary_mask.cpu().numpy())
        if iter % 5 == 0:
            print('\riteration: [%d/%d] Score: %.2f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, o_score, lr, loss.data.item(),
                res_time), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    sc_scores = sc_evaluation.get_scores()
    bc_scores = bc_evaluation.get_scores()

    return average_epoch_loss_train, sc_scores, bc_scores, lr


def train_val_change_detection(args):
    torch.backends.cudnn.benchmark = True
    SEED = 3047
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if args.file_name == 'SECOND':
        args.data_root = '/data2/ywj/SCD/json/SECOND'
        args.num_classes = 7
    elif args.file_name == 'LandsatSCD':
        args.data_root = '/data2/ywj/SCD/json/LandsatSCD'
        args.num_classes = 5
    elif args.file_name == 'WHU':
        args.data_root = '/data/yjy/FreeStyleNet/outputs/WHU_LIS_AB_From_SCD-1-10'
        args.num_classes = 2
    elif args.file_name == 'SyntheticID':
        args.data_root = '/data/yrz/repos/SCD/data/SECOND'
        args.num_classes = 7
    elif args.file_name == 'CNAM-CD_V1':
        args.data_root = '/data2/ywj/SCD/json/CNAM-CD_V1'
        args.num_classes = 6
    elif args.file_name == 'GF':
        args.data_root = '/data2/ywj/SCD/json/GF'
        args.num_classes = 9
    else:
        raise TypeError('%s has not defined' % args.file_name)
    assert args.max_epochs or args.max_steps
    if args.max_epochs:
        args.save_dir = args.save_dir + args.model_name + '/' + args.file_name + '_epoch_' + str(
            args.max_epochs) + '_lr_' + str(args.lr) + '/'
    elif args.max_steps:
        args.save_dir = args.save_dir + args.model_name + '/' + args.file_name + '_iter_' + str(
        args.max_steps) + '_lr_' + str(args.lr) + '/'
    else:
        raise ValueError
    if not args.json_file:
        json_file = 'train.json'
    else:
        json_file = args.json_file
    args.save_dir += os.path.basename(json_file) + '/'
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = get_model_by_name(args.model_name, args.num_classes, args.inWidth)
    model = model.cuda()


    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    total_params = total_params / 1e6
    print('Total network parameters (excluding idr): ' + str(total_params))
    total_params_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_to_update = total_params_to_update / (1024 * 1024)
    print('Total parameters to update: ' + str(total_params_to_update))

    train_data = myDataLoader.Dataset('train', file_name=args.file_name, data_root=args.data_root, transform=True, json_file=json_file, mode='train')
    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    test_data = myDataLoader.Dataset("val", file_name='SECOND', data_root='/data2/ywj/SCD/json/SECOND', transform=False,json_file='val_ori_SECOND_Image.json', mode='test')
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    max_batches = len(trainLoader)
    print('For each epoch, we have {} batches'.format(max_batches))
    if not args.max_epochs:
        args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_value = 0

    logFileLoc = args.save_dir + args.logFile
    column_width = 12  # Adjust this width based on your preference
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Total network parameters: %.2f" % total_params)
        logger.write("\nTotal parameters to update: %.2f" % total_params_to_update)
        header = "\n{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|" \
                 "{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}".format(
            'Epoch', 'OA (sc)', 'Score (sc)', 'mIoU (sc)', 'Sek (sc)', 'Fscd (sc)',
            'Kappa (bc)', 'IoU (bc)', 'F1 (bc)', 'Rec (bc)', 'Pre (bc)', width=column_width
        )
        logger.write(header)
    logger.flush()

    alpha = [1.] * args.num_classes
    alpha = torch.as_tensor(alpha).contiguous().cuda()
    criterion_sc = MutilCrossEntropyLoss(alpha=alpha, ignore_index=0).cuda()
    criterion_bc = DeepSupervisionLoss().cuda()
    criterion_con = ChangeSimilarity().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, (0.9, 0.999), weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(start_epoch, args.max_epochs):
        loss_tr, sc_score_tr, bc_score_tr, lr = train(args, trainLoader, model, criterion_sc, criterion_bc,
                                                      criterion_con, scaler, optimizer, max_batches, cur_iter)
        cur_iter += len(trainLoader)
        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': loss_tr,
            'ScoreTr': sc_score_tr['Score'],
            'lr': lr
        }, args.save_dir + 'checkpoint_not.pth.tar')

        loss_val, sc_score_val, bc_score_val = validate(args, testLoader, model, criterion_sc, criterion_bc,
                                                        criterion_con)

        val_line = "\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|" \
                   "{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}".format(
            epoch, sc_score_val['OA'], sc_score_val['Score'], sc_score_val['mIoU'], sc_score_val['Sek'],
            sc_score_val['Fscd'], bc_score_val['Kappa'], bc_score_val['IoU'], bc_score_val['F1'],
            bc_score_val['recall'], bc_score_val['precision'], width=column_width
        )
        logger.write(val_line)
        logger.flush()

        # save the model also
        model_file_name = args.save_dir + 'best_model_not.pth'
        if epoch % 1 == 0 and max_value <= sc_score_val['Score']:
            max_value = sc_score_val['Score']
            torch.save(model.state_dict(), model_file_name)

        # if epoch==args.max_epochs-1:
        #     torch.save(model.state_dict(), model_file_name_final)

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.2f\tVal Loss = %.2f\t Score(tr) = %.2f\t Score(val) = %.2f"
              % (epoch, loss_tr, loss_val, sc_score_tr['Score'], sc_score_val['Score']))
        torch.cuda.empty_cache()

    # model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, sc_score_test, bc_score_test = validate(args, testLoader, model, criterion_sc, criterion_bc,
                                                       criterion_con)
    print("\nTest :\t OA (te) = %.2f\t mIoU (te) = %.2f\t Sek (te) = %.2f\t Fscd (te) = %.2f"
          % (sc_score_test['OA'], sc_score_test['mIoU'], sc_score_test['Sek'], sc_score_test['Fscd']))
    test_line = "\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|" \
                "{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}".format(
        'Test', sc_score_test['OA'], sc_score_test['Score'], sc_score_test['mIoU'], sc_score_test['Sek'],
        sc_score_test['Fscd'], bc_score_test['Kappa'], bc_score_test['IoU'], bc_score_test['F1'],
        bc_score_test['recall'], bc_score_test['precision'], width=column_width
    )
    logger.write(test_line)
    logger.flush()
    logger.close()


if __name__ == '__main__':
    """
    json_dir='PSDImage_label_1-10'
    json_name='train_concat_from_FreeStyle_14_rule5'
    CUDA_VISIBLE_DEVICES=2 nohup python train.py --model_name A2Net  \
                                            --file_name SECOND \
                                            --inWidth 512 --inHeight 512 \
                                            --save_dir ./weights/ \
                                            --lr 5e-4 --batch_size 8 --max_epochs 50 \
                                            --json_file ${json_dir}/${json_name}.json > logs/${json_dir}/${json_name}_epoch50.file 2>&1 &
    """

    parser = ArgumentParser()
    parser.add_argument('--model_name', default="SCanNet", help='Data directory')
    parser.add_argument('--file_name', default="SECOND", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=0, help='Max. number of iterations')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='SCanNet_ori_SECOND_add_Syn_not_bin.txt', help='File that stores the training and validation logs')
    parser.add_argument('--json_file', default='FreeStyleNet_Syn_SECOND_add_100_ori.json',help='the json file path, you just should input json name')
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_val_change_detection(args)
