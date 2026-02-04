import argparse
import os.path
import warnings
from torch.utils.data import DataLoader
import pprint
from utils.helper import *
import torchvision.transforms as transforms
from datasets import AnomalyClassificationDataset, FixedInBatchSampler
from utils.metrics import compute_pixelwise_metrics, compute_imagewise_metrics, compute_iou_metrics
from utils.losses import ADloss
import torch.nn as nn
from discriminator import Discriminator
import torch.nn.functional as F

warnings.filterwarnings('ignore')

def main(args):

    set_seed(args.seed)
    args.category = "_".join(args.sample_file.split('_')[1:-1])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.save_root = os.path.join(args.experiment_root, args.dataset, args.category,
                                  "{}_shot_discriminator".format(args.sample_file.split('_')[0]))

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    logger = create_logger("logger", os.path.join(args.save_root, 'logger_{}.log'.format(args.category)))
    logger.info("config: {}".format(pprint.pformat(args)))

    test_dataset = AnomalyClassificationDataset(data_root=args.data_root,
                                                experiment_root=args.experiment_root,
                                                dataset = args.dataset,
                                                category=args.category,
                                                split='test',
                                                sample_file=None,
                                                use_gen = False,
                                                image_size=args.image_size,
                                                transforms = transform)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False, num_workers=4)

    class_list = test_dataset.class_list
    logger.info("labels: {}".format(pprint.pformat(class_list)))

    train_dataset = AnomalyClassificationDataset(data_root = args.data_root,
                                                 experiment_root = args.experiment_root,
                                                 dataset = args.dataset,
                                                 category = args.category,
                                                 sample_file = args.sample_file,
                                                 image_size = args.image_size,
                                                 use_gen = args.usegen,
                                                 class_list = class_list,
                                                 split ='train',
                                                 transforms = transform)

    batch_sampler = FixedInBatchSampler(train_dataset,
                                        args.batch_size,
                                        args.num_steps_per_epoch,
                                        is_zero_shot = args.sample_file.startswith('zero'))

    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)


    vision_pretrained_weights = os.path.join(args.ckpt_root, 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    text_pretrained_weights = os.path.join(args.ckpt_root, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth')

    model = Discriminator(
            labels = [label[0] for label in class_list[1:]],
            vision_pretrained_weights= vision_pretrained_weights,
            text_pretrained_weights = text_pretrained_weights,
            device = device)

    model.to(device)

    params_list = [
        {"params": model.necks.parameters(), "lr": args.base_lr},
        {"params": model.fusionnets.parameters(), "lr": args.base_lr},
        {"params": model.epsilon, "lr": args.base_lr},
    ]

    optimizer = torch.optim.AdamW(
            params_list,
            lr=args.base_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: pow(1.0 - x / args.epochs, 0.9))
    best_result = None

    for epoch in range(0, args.epochs):

        last_iter = epoch * len(train_dataloader)

        train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            last_iter,
            model,
            device,
        )

        scheduler.step()

        results = validate(args, test_dataloader, epoch, model, device, class_list)

        results_record = {item: results[item] for item in results}
        record = Report(['']+[item for item in results_record])
        record.add_one_record(['Epoch {}'.format(epoch+1)] + [results_record[item] for item in results_record])

        if best_result is None:
            best_result = results
            save_checkpoint(model, os.path.join(args.save_root, 'best_ckpt.pkl'))
        else:
            if sum([best_result[key] for key in best_result]) < sum([results_record[item] for item in results_record]):
                best_result = results
                save_checkpoint(model, os.path.join(args.save_root, 'best_ckpt.pkl'))

        best_results_record = {item: best_result[item] for item in best_result if item.find('threshold') == -1}
        record.add_one_record(['best']+[best_results_record[item] for item in best_results_record])
        logger.info(f"\n{record}")

    save_checkpoint(model, os.path.join(args.save_root, 'latest_ckpt.pkl'))


def train_one_epoch(
            args,
            train_dataloader,
            optimizer,
            epoch,
            start_iter,
            model,
            device,
):

    logger = logging.getLogger("logger")

    loss_meter = AverageMeter(args.print_freq_step)
    focal_meter = AverageMeter(args.print_freq_step)
    ce_meter = AverageMeter(args.print_freq_step)
    dice_meter = AverageMeter(args.print_freq_step)

    criterion = ADloss(ignore_index=-1)

    model.train()
    model.vision_encoder.eval()
    model.text_encoder.eval()

    for i, input in enumerate(train_dataloader):
        curr_step = start_iter + i

        images = input['image'].to(device)
        cls_masks = input['cls_mask'].to(device)
        dt_masks = input['dt_mask'].to(device)
        is_synthesize = input['is_synthesize'].to(device)
        assert torch.all(input['is_combined'] == False)

        detection_maps, classification_maps = model(images)

        focal_loss, dice_loss, ce_loss = criterion(detection_maps,
                                                      dt_masks,
                                                      classification_maps,
                                                      cls_masks,
                                                      is_synthesize)

        focal_meter.update(focal_loss.item())
        ce_meter.update(ce_loss.item())
        dice_meter.update(dice_loss.item())

        loss = focal_loss + dice_loss + ce_loss * 0.5
        loss_meter.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.fusionnets.parameters(), max_norm=0.1)
        optimizer.step()

        if (curr_step + 1) % args.print_freq_step == 0:
            logger.info(
                "Epoch: [{0}/{1}]\t"
                "Iter: [{2}/{3}]\t"
                "Lr [{4:.6f}/{5:.6f}]\t"
                "Focal Loss {focal_loss.val:.5f} ({focal_loss.avg:.5f})\t"
                "Dice Loss {dice_loss.val:.5f} ({dice_loss.avg:.5f})\t"
                "CE Loss {ce_loss.val:.5f} ({ce_loss.avg:.5f})\t"
                "Loss {loss.val:.5f} ({loss.avg:.5f})"
                    .format(
                    epoch+1 ,
                    args.epochs,
                    curr_step + 1,
                    len(train_dataloader) * args.epochs,
                    optimizer.param_groups[0]['lr'],
                    args.base_lr,
                    focal_loss=focal_meter,
                    dice_loss=dice_meter,
                    ce_loss=ce_meter,
                    loss = loss_meter,
                )
            )


@torch.no_grad()
def validate(args, test_dataloader, epoch, model, device, class_list):

    model.eval()

    dt_preds = []
    dt_gts = []
    cls_preds = []
    cls_gts = []

    anomaly_labels = []
    image_paths = []
    combined_labels = []

    for input in test_dataloader:

        images = input['image'].to(device)
        dt_maps, cls_maps = model(images)

        class_mask = input['cls_mask']
        dt_mask = input['dt_mask']

        dt_preds.append(dt_maps.cpu())
        cls_preds.append(cls_maps.cpu())

        dt_gts.append(dt_mask)
        cls_gts.append(class_mask)

        anomaly_labels.append(input['anomaly_type'])
        combined_labels.append(input['is_combined'])
        image_paths.extend(input['paths'])

    dt_preds = torch.cat(dt_preds, dim=0)
    dt_gts = torch.cat(dt_gts, dim=0)

    if dt_preds.ndim == 4:
        dt_preds = dt_preds.squeeze(1)

    results = {}

    dt_preds = F.sigmoid(dt_preds)
    image_level_dt_preds = get_image_level_score(dt_preds)
    image_level_dt_gts = torch.max(dt_gts.view((dt_gts.size(0), -1)), dim=-1)[0]

    results.update(compute_imagewise_metrics(image_level_dt_preds.numpy(), image_level_dt_gts.numpy()))
    results.update(compute_pixelwise_metrics(dt_preds.numpy(), dt_gts.numpy()))

    cls_preds = torch.cat(cls_preds, dim=0)
    cls_gts = torch.cat(cls_gts, dim=0)
    anomaly_labels = torch.cat(anomaly_labels, dim=0)
    combined_labels = torch.cat(combined_labels, dim=0)

    dt_preds = norm_func(dt_preds)
    anomaly_class_preds = torch.max(cls_preds, 1)[1].cpu()
    anomaly_class_preds += 1
    anomaly_class_preds[dt_preds < args.threshold] = 0

    cls_gts_wo_combined = cls_gts[~combined_labels].numpy() + 1
    anomaly_class_preds_wo_combined = anomaly_class_preds[~combined_labels].numpy()

    results.update(compute_iou_metrics(anomaly_class_preds_wo_combined, cls_gts_wo_combined, num_classes=len(class_list)))

    image_level_cls_preds = []
    for anomaly_class in anomaly_class_preds:
        if torch.any(anomaly_class != 0):
            anomaly_points = anomaly_class[anomaly_class != 0]
            unique_vals, val_counts = torch.unique(anomaly_points, return_counts=True)
            image_level_cls_preds.append(int(unique_vals[torch.argmax(val_counts)].item()))
        else:
            image_level_cls_preds.append(0)

    anomaly_labels_wo_combined = anomaly_labels[~combined_labels].numpy()
    results.update({"acc": np.mean((np.array(image_level_cls_preds)[~(combined_labels.numpy())] == anomaly_labels_wo_combined).astype(float))})
    return results


def save_checkpoint(model, save_path):
    save_params = {
        "necks": model.necks.state_dict(),
        "fusionnets": model.fusionnets.state_dict(),
        "epsilon": model.epsilon
    }
    torch.save(save_params, save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train multi-task discriminator")
    parser.add_argument("--data_root", type=str, default='data')
    parser.add_argument("--dataset", type=str, default='mvtec')
    parser.add_argument("--sample_file", type=str, default='zero_hazelnut_n2.jsonl')
    parser.add_argument("--experiment_root", default="experiment", type=str)
    parser.add_argument("--ckpt_root", default="ckpt", type=str)

    parser.add_argument("--threshold", type=float, default = 0.5)
    parser.add_argument("--base_lr", type=float, default = 0.001)
    parser.add_argument("--image_size", type=int, default = 512)
    parser.add_argument("--batch_size", type=int, default = 16)

    parser.add_argument("--epochs", type=int, default = 50)
    parser.add_argument("--num_steps_per_epoch", type=int, default= 5)
    parser.add_argument("--print_freq_step", type=int, default= 1)
    parser.add_argument("--seed", type=int, default = 10)
    parser.add_argument('--usegen', action='store_true', default='whether to use synthetic samples for training')

    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)
