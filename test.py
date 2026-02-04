import argparse
import os.path
import warnings
import torch
from torch.utils.data import DataLoader
from utils.helper import *
import torchvision.transforms as transforms
from datasets import AnomalyClassificationDataset
from utils.metrics import compute_pixelwise_metrics, compute_imagewise_metrics, compute_iou_metrics
from discriminator import Discriminator
import torch.nn.functional as F
warnings.filterwarnings('ignore')


def main(args):
    set_seed(args.seed)
    args.dataset = args.checkpoint_path.split('/')[1]
    args.category = args.checkpoint_path.split('/')[2]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    test_dataset = AnomalyClassificationDataset(data_root=args.data_root,
                                                experiment_root=None,
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

    vision_pretrained_weights = os.path.join(args.ckpt_root, 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    text_pretrained_weights = os.path.join(args.ckpt_root, 'dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth')

    model = Discriminator(
            labels=[label[0] for label in class_list[1:]],
            vision_pretrained_weights=vision_pretrained_weights,
            text_pretrained_weights=text_pretrained_weights,
            device=device)

    loaded_params = torch.load(args.checkpoint_path)
    model.necks.load_state_dict(loaded_params['necks'])
    model.fusionnets.load_state_dict(loaded_params['fusionnets'])
    model.epsilon.data = loaded_params['epsilon']
    model.to(device)
    model.eval()

    results = evaluate(args, test_dataloader, model, device, class_list, save_root=os.path.dirname(args.checkpoint_path))

    results_record = {item: results[item] for item in results if item.find('threshold') == -1}
    record = Report(['']+[item for item in results_record])
    record.add_one_record(['evaluate'] + [results_record[item] for item in results_record])
    print(f"\n{record}")


@torch.no_grad()
def evaluate(args, test_dataloader, model, device, class_list, save_root):

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

    dt_preds = F.sigmoid(dt_preds)

    results = {}
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

    results.update(compute_iou_metrics(anomaly_class_preds_wo_combined,
                                       cls_gts_wo_combined, num_classes=len(class_list)))

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

    new_palette = get_new_pallete(len(class_list))


    for image_path, cls_gt, detection_pred, class_pred in zip(image_paths, cls_gts, dt_preds.numpy(), anomaly_class_preds.numpy()):

        image = Image.open(image_path).convert("RGB")
        image = np.array(image.resize((detection_pred.shape[0], detection_pred.shape[1]))).astype(np.uint8)
        detection_image = show_cam_on_image(image / 255, detection_pred, use_rgb=True)

        palette_image = get_new_mask_pallete(class_pred, new_palette)
        palette_image = palette_image.convert("RGB")

        cls_gt = cls_gt.cpu().numpy()

        cls_image = get_new_mask_pallete(cls_gt + 1, new_palette)
        cls_image = cls_image.convert("RGB")

        merge_image = np.concatenate([image, cls_image, detection_image, np.array(palette_image)], axis=1)
        merge_image = Image.fromarray(merge_image.astype(np.uint8))

        image_root, image_name = os.path.split(image_path)
        _, sub_class = os.path.split(image_root)
        image_save_root = os.path.join(save_root, 'vis', sub_class)
        os.makedirs(image_save_root, exist_ok=True)

        merge_image.save(os.path.join(image_save_root, image_name))

    return results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load Checkpoint")
    parser.add_argument("--data_root", type=str, default='data')
    parser.add_argument("--ckpt_root", default="ckpt", type=str)
    parser.add_argument("--threshold", type=float, default = 0.5)
    parser.add_argument("--image_size", type=int, default = 512)
    parser.add_argument("--batch_size", type=int, default = 16)
    parser.add_argument("--seed", type=int, default = 10)
    parser.add_argument("--checkpoint_path", type=str, required=True, help='checkpoint path')
    args = parser.parse_args()
    torch.multiprocessing.set_start_method("spawn")
    main(args)