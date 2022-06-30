import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
import random
from utils import AverageMeter, calculate_accuracy, write_to_batch_logger, write_to_epoch_logger

from graph_configure import *
def train_epoch(epoch,
                data_loader,
                model,
                criterion,
                optimizer,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                tb_writer=None,
                distributed=False):
    print('train at epoch {}'.format(epoch))
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs, features = model(inputs)

        loss = criterion(outputs, targets)
        acc = calculate_accuracy(outputs, targets)

        losses.update(loss.item(), inputs.size(0))
        accuracies.update(acc, inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(1)
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss=losses,
                     acc=accuracies), flush=True)

    if distributed:
        loss_sum = torch.tensor([losses.sum], dtype=torch.float32, device=device)
        loss_count = torch.tensor([losses.count], dtype=torch.float32, device=device)
        acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
        acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    write_to_epoch_logger(epoch_logger, epoch, losses.val, accuracies.val, current_lr)

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', losses.avg, epoch)
        tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


def train_eeg_epoch(epoch,
                  data_loader,
                  model,
                  joint_prediction_eeg,
                  criterion,
                  criterion_jsd,
                  criterion_Khn_eeg,
                  optimizer,
                  optimizer_eeg,
                  device,
                  current_lr,
                  epoch_logger,
                  batch_logger,
                  tb_writer=None,
                  distributed=False):
    print('train at epoch {}'.format(epoch))
    model.train()
    joint_prediction_eeg.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    # classification loss
    losses_cls = AverageMeter()
    accuracies = AverageMeter()
    # contrastive loss
    losses_khn_eeg = AverageMeter()
    # jsd loss
    losses_jsd_a = AverageMeter()
    lossse_sta= AverageMeter()


    end_time = time.time()

    for i, (inputs, targets, eegs) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        outputs, features = model(inputs)             # model -r2+1d
        targets = targets.to(device, non_blocking=True)
        eegs = eegs.to(device, non_blocking=True)                #16*512


        loss_cls_v = criterion(outputs, targets) # video classification loss
        acc = calculate_accuracy(outputs, targets)
        #####################################################################################
        # use audio features as features & filter out the zero-ones (not available) audio features
        features_eeg = eegs[eegs.sum(dim=1) != 0]
        features_vid = features[eegs.sum(dim=1) != 0]
        targets_new = targets[eegs.sum(dim=1) != 0]
        outputs_new = outputs[eegs.sum(dim=1) != 0]
        %import pdb
        %pdb.set_trace()
        # here compose images and videos

        outputs_eeg, features_eeg = joint_prediction_eeg(features_eeg, features_vid)#都是16*512
        loss_cls_a = criterion(outputs_eeg, targets_new) # video classification loss


       #####################

        #output_
        #graph_match_loss=1





       #####################
        # contrastive learning (symmetric loss)
        loss_vm = criterion_khn_eeg(features_vid, features_eeg, targets_new) + criterion_khn_eeg(features_eeg, features_vid, targets_new)
        # align video features to eeg features
        loss_va = criterion_khn_eeg(features_vid, features_eeg, targets_new) + criterion_khn_eeg(features_eeg, features_vid, targets_new)
        # align multimodal (audio-video) features to audio features
        loss_ma = criterion_khn_eeg(features_eeg, features_eeg, targets_new) + criterion_khn_eeg(features_eeg, features_eeg, targets_new)
        # contrastive loss
        loss_khn_eeg = loss_vm + loss_va
        loss_khn_a = loss_vm + loss_ma
        # jsd loss
        loss_jsd_a = criterion_jsd(outputs_new, outputs_eeg)
        #####################################################################################
        total_loss_v = sum([loss_cls_v, loss_khn_eeg, loss_jsd_a])
        total_loss_a = sum([loss_cls_a, loss_khn_a, loss_jsd_a])

        losses_cls.update(loss_cls_v.item(), inputs.size(0))
        losses_khn_eeg.update(loss_khn_eeg.item(), inputs.size(0))
        losses_jsd_a.update(loss_jsd_a.item(), inputs.size(0))

        accuracies.update(acc, inputs.size(0))

        optimizer_eeg.zero_grad()
        total_loss_a.backward(retain_graph=True)

        optimizer.zero_grad()
        total_loss_v.backward()

        optimizer_eeg.step()
        optimizer.step()
        print(0)
        #####################################################################################
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        write_to_batch_logger(batch_logger, epoch, i, data_loader, losses_cls.val, accuracies.val, current_lr)

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_cls {loss_cls.val:.3f} ({loss_cls.avg:.3f})\t'
              'Loss_khn_a {loss_khn_av.val:.3f} ({loss_khn_av.avg:.3f})\t'
              'Loss_jsd_a {loss_jsd_a.val:.3f} ({loss_jsd_a.avg:.3f})\t'
              'Acc {acc.val:.3f} ({acc.avg:.3f})'.
              format(epoch, i + 1, len(data_loader),
                     batch_time=batch_time,
                     data_time=data_time,
                     loss_cls=losses_cls,
                     loss_khn_eeg=losses_khn_eeg,
                     loss_jsd_a=losses_jsd_a,
                     acc=accuracies), flush=True)

        if distributed:
            loss_cls_sum = torch.tensor([losses_cls.sum], dtype=torch.float32, device=device)
            loss_khn_av_sum = torch.tensor([losses_khn_av.sum], dtype=torch.float32, device=device)
            loss_jsd_a_sum = torch.tensor([losses_jsd_a.sum], dtype=torch.float32, device=device)
            acc_sum = torch.tensor([accuracies.sum], dtype=torch.float32, device=device)
            loss_count = torch.tensor([losses_cls.count], dtype=torch.float32, device=device)
            acc_count = torch.tensor([accuracies.count], dtype=torch.float32, device=device)

            dist.all_reduce(loss_cls_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_khn_av_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_jsd_a_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
            dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

            losses_cls.avg = loss_cls_sum.item() / loss_count.item()
            losses_khn_av.avg = loss_khn_av_sum.item() / loss_count.item()
            losses_jsd_a.avg = loss_jsd_a_sum.item() / loss_count.item()
            accuracies.avg = acc_sum.item() / acc_count.item()

        write_to_epoch_logger(epoch_logger, epoch, losses_cls.val, accuracies.val, current_lr)

        if tb_writer is not None:
            tb_writer.add_scalar('train/loss_cls', losses_cls.avg, epoch)
            tb_writer.add_scalar('train/loss_khn_eeg', losses_khn_eeg.avg, epoch)
            tb.writer.add_scalar('train/loss_sta', losses_sta.avg,epoch)
            tb_writer.add_scalar('train/loss_jsd_a', losses_jsd_a.avg, epoch)
            tb_writer.add_scalar('train/acc', accuracies.avg, epoch)


