import numpy as np
import pandas as pd
import time
import timeit
from datetime import datetime
import os
import glob
import natsort
import shutil
from tqdm.notebook import tqdm
import re
import sys
import cv2
from PIL import Image
import random
import copy
import importlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
from torchvision import datasets
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader 
from monai.losses import TverskyLoss as TverskyLoss
import albumentations as A

def control_random_seed(seed, pytorch=True):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available()==True:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
        torch.backends.cudnn.benchmark = False
def imread_kor ( filePath, mode=cv2.IMREAD_UNCHANGED ) : 
    stream = open( filePath.encode("utf-8") , "rb") 
    bytes = bytearray(stream.read()) 
    numpyArray = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpyArray , mode)
def imwrite_kor(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result:
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
                return True
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False
    

def random_rotation(image, mask, angle_range=(-30, 30)):
    # 지정된 각도 범위 내에서 무작위로 각도 선택
    angle = random.uniform(angle_range[0], angle_range[1])
    # 이미지와 마스크를 동일한 각도로 회전
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)
    return image, mask
class ImageTransforms:
    def __init__(self, image, mask=None, in_channels=3):
        self.image = image
        self.mask = mask
        self.in_channels = in_channels

    def HWR(self, h, w):
        # 현이미지와 동일한 비율을 유지하며 (h, w) 안에 들어갈 수 있는 최대 크기로 리사이징
        orig_h, orig_w = self.image.shape[:2]
        scale = min(h / orig_h, w / orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        self.image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return self

    def P(self, h, w):
        # 이미지에 zero-padding을 추가하여 (h, w) 크기로 만듦
        pad_h = max(0, h - self.image.shape[0])
        pad_w = max(0, w - self.image.shape[1])
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if self.in_channels == 1:
            value = 0
        else:
            value = [0] * self.in_channels
        self.image = cv2.copyMakeBorder(self.image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=value)
        if self.mask is not None:
            self.mask = cv2.copyMakeBorder(self.mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        return self

    def R(self, h, w):
        # 이미지를 (h, w) 크기로 리사이즈
        self.image = cv2.resize(self.image, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return self

    def SQP(self):
        # 정사각형으로 만들기 위해 짧은 축 기준으로 zero-padding
        h, w = self.image.shape[:2]
        size = max(h, w)
        self.P(size, size)
        return self
    def C(self, h, w):
        # 이미지를 (h, w) 크기로 랜덤 크롭핑
        img_h, img_w = self.image.shape[:2]
        if img_h < h or img_w < w:
            raise ValueError("크롭 크기가 이미지 크기보다 큽니다. 먼저 패딩 또는 리사이징을 수행하세요.")
        
        top = np.random.randint(0, img_h - h + 1)
        left = np.random.randint(0, img_w - w + 1)
        
        self.image = self.image[top:top + h, left:left + w]
        if self.mask is not None:
            self.mask = self.mask[top:top + h, left:left + w]
        return self
    def get_image_and_mask(self):
        return self.image, self.mask

    def get_image_and_mask(self):
        return self.image, self.mask

class ImagesDataset(Dataset):
    def __init__(self, image_path_list, target_path_list):
        self.image_path_list = image_path_list
        self.target_path_list = target_path_list
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.image_path_list)

    def parse_transform_str(self, image, mask):
        in_channels = 1 if len(image.shape) == 2 else image.shape[2]
        transform = ImageTransforms(image, mask, in_channels)
        return transform.get_image_and_mask()

    def __getitem__(self, idx):
        image_path = self.image_path_list[idx]
        mask_path = self.target_path_list[idx]
        if os.path.splitext(os.path.basename(image_path))[0] != os.path.splitext(os.path.basename(mask_path))[0]:
            print("Filenames don't match")
            
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        mask = np.load(mask_path)
        mask = mask.astype(np.uint8)  # Ensure mask is in the correct format

        # transformed_image, transformed_mask = self.parse_transform_str(image, mask)
      
        image = self.transform(image.copy()).float()
        mask = torch.tensor(mask).long().unsqueeze(0)
        return image, mask, image_path

class SegDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, fill_last_batch=False):
        self.fill_last_batch = fill_last_batch
        self.dataset = dataset
        super().__init__(dataset, batch_size, shuffle, sampler,
                         batch_sampler, num_workers, collate_fn,
                         pin_memory, drop_last)

    def __iter__(self):
        batch_iter = super().__iter__()
        for batch in batch_iter:
            if self.fill_last_batch and len(batch[0]) < self.batch_size:
                additional_samples_needed = self.batch_size - len(batch[0])
                additional_indices = random.choices(range(len(self.dataset)), k=additional_samples_needed)
                additional_samples = [self.dataset[idx] for idx in additional_indices]

                if isinstance(batch, (list, tuple)):
                    batch = list(batch)
                    for i in range(len(batch) - 1):  # Process tensor elements (image, mask)
                        batch[i] = torch.cat([batch[i], torch.stack([sample[i] for sample in additional_samples])])
                    batch[-1] = batch[-1] + [sample[-1] for sample in additional_samples]  # Process string elements (image paths)
                    batch = tuple(batch)
                else:
                    batch = torch.cat([batch, torch.stack(additional_samples)])
            yield batch
# 클래스별 고유 색상 지정 (클래스 0은 배경, 나머지는 각 클래스에 고유 색상)
class_colors = {
    0: [0, 0, 0],         # 클래스 0: 검정색 (배경)
    1: [255, 0, 0],       # 클래스 1: 빨강
    2: [0, 255, 0],       # 클래스 2: 초록
    3: [0, 0, 255],       # 클래스 3: 파랑
    4: [255, 255, 0],     # 클래스 4: 노랑
    5: [255, 0, 255],     # 클래스 5: 분홍
    6: [0, 255, 255],     # 클래스 6: 하늘색
    7: [128, 0, 128],     # 클래스 7: 보라색
    8: [128, 128, 0],     # 클래스 8: 올리브색
    9: [0, 128, 128],     # 클래스 9: 청록색
    10: [128, 128, 128],  # 클래스 10: 회색
    11: [192, 0, 0],      # 클래스 11: 진한 빨강
    12: [0, 192, 0],      # 클래스 12: 진한 초록
    13: [0, 0, 192],      # 클래스 13: 진한 파랑
    14: [192, 192, 0],    # 클래스 14: 연한 노랑
    15: [192, 0, 192],    # 클래스 15: 연한 분홍
    16: [0, 192, 192],    # 클래스 16: 연한 하늘색
    17: [128, 64, 0],     # 클래스 17: 갈색
    18: [64, 128, 0],     # 클래스 18: 연두색
    19: [0, 64, 128],     # 클래스 19: 어두운 청록색
    20: [255, 128, 0],    # 클래스 20: 주황색
    21: [128, 0, 255],    # 클래스 21: 보라색 (밝은)
    22: [0, 128, 255],    # 클래스 22: 밝은 파랑
    23: [255, 128, 128],  # 클래스 23: 연한 빨강
    24: [128, 255, 128],  # 클래스 24: 연한 초록
    25: [128, 128, 255],  # 클래스 25: 연한 파랑
    26: [255, 255, 128],  # 클래스 26: 연한 노랑
    27: [255, 0, 128],    # 클래스 27: 핫핑크
    28: [128, 255, 0],    # 클래스 28: 형광 초록
    29: [0, 255, 128],    # 클래스 29: 밝은 청록색
}

    
def colorize_mask(mask):
    # 마스크에 색상 입히기
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for cls, color in class_colors.items():
        color_mask[mask == cls] = color
    return color_mask
    
class DiceCELoss:
    """
    Dice Loss와 CE Loss의 결합 손실 클래스
    """
    def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
        """
        Args:
            weight (float): Dice Loss와 BCE Loss 사이의 가중치 (0~1 사이의 값)
            epsilon (float): 0으로 나누는 것을 방지하기 위한 작은 값
            mode (str): 'binary' 또는 'multiclass'로 손실 계산 모드를 설정
        """
        self.weight = weight
        self.epsilon = epsilon
        self.mode = mode
    
    def __call__(self, pred, target):
        """
        결합 손실 계산 함수
        
        Args:
            pred (torch.Tensor): 예측된 확률맵, 
                - binary segmentation: shape이 (batchsize, 1, H, W)
                - multiclass segmentation: shape이 (batchsize, num_classes, H, W)
            target (torch.Tensor): 정답 마스크, shape이 (batchsize, 1, H, W)
            
        Returns:
            torch.Tensor: 계산된 결합 손실 값
        """
        if self.mode == 'binary':
            # Binary Dice Loss 계산
            pred = pred.squeeze(1)  # shape: (batchsize, H, W)
            target = target.squeeze(1).float()
            intersection = torch.sum(pred * target, dim=(1, 2))
            union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            # BCE Loss 계산
            ce_loss = F.binary_cross_entropy(pred, target)
        
        elif self.mode == 'multiclass':
            # Multiclass Dice Loss 계산
            batchsize, num_classes, H, W = pred.shape
            target = target.squeeze(1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
            intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
            union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            # Cross Entropy Loss 계산
            ce_loss = F.cross_entropy(pred, target)
        else:
            raise ValueError("mode should be 'binary' or 'multiclass'")
        
        # 결합 손실 계산
        combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
        
        return combined_loss

class DiceCoefficient:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        y_pred_one_hot = np.eye(self.num_classes)[y_pred]
        intersection = np.sum(y_true_one_hot * y_pred_one_hot, axis=(1, 2))
        union = np.sum(y_true_one_hot, axis=(1, 2)) + np.sum(y_pred_one_hot, axis=(1, 2))
        dice = (2. * intersection) / (union + 1e-6)
        return dice

class IoU:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        y_pred_one_hot = np.eye(self.num_classes)[y_pred]
        intersection = np.sum(y_true_one_hot* y_pred_one_hot, axis=(1, 2))
        union = np.sum(y_true_one_hot, axis=(1, 2)) + np.sum(y_pred_one_hot, axis=(1, 2)) - intersection
        iou = intersection / (union + 1e-6)
        return iou

class Precision:
    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        y_pred_one_hot = np.eye(self.num_classes)[y_pred]
        tp = np.sum(y_true_one_hot* y_pred_one_hot, axis=(1, 2))
        fp = np.sum((y_pred_one_hot == 1) & (y_true_one_hot== 0), axis=(1, 2))
        precision = tp / (tp + fp + 1e-6)
        return precision

class Recall:
    def __init__(self,  num_classes=None):
        self.num_classes = num_classes

    def __call__(self, y_pred, y_true):
        y_true_one_hot = np.eye(self.num_classes)[y_true]
        y_pred_one_hot = np.eye(self.num_classes)[y_pred]
        tp = np.sum(y_true_one_hot* y_pred_one_hot, axis=(1, 2))
        fn = np.sum((y_true_one_hot == 1) & (y_pred_one_hot == 0), axis=(1, 2))
        recall = tp / (tp + fn + 1e-6)
        return recall
    
def check_class_presence(batch_masks, num_classes):
    """
    각 샘플이 각 클래스에 대해 positive를 갖는지 체크하는 함수 (binary, multi class 동일)

    Args:
        batch_masks (numpy.ndarray): shape이 (batchsize, H, W)인 멀티 클래스 GT 마스크 파일
        num_classes (int): 클래스의 수 (0부터 num_classes-1까지의 레이블을 가짐)

    Returns:
        numpy.ndarray: shape이 (batchsize, num_classes)인 Boolean 배열로,
                       각 샘플이 각 클래스에 대해 positive를 가지면 True, 그렇지 않으면 False
    """
    batchsize = batch_masks.shape[0]
    presence_matrix = np.zeros((batchsize, num_classes), dtype=bool)
    for i in range(batchsize):
        for c in range(num_classes):
            presence_matrix[i, c] = np.any(batch_masks[i] == c)

    return presence_matrix

def train(train_loader, epoch, model, criterion, optimizer, device, activation ):
    model.train()
    train_losses=AverageMeter()

    for i, (input, target, _) in enumerate(tqdm(train_loader, desc="Training", unit="batch", leave=False)):
        input = input.to(device)
        target = target.to(device)

        output = activation(model(input)) 
        loss = criterion(output,target).float()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.detach().cpu().numpy(),input.shape[0])

    Train_Loss=np.round(train_losses.avg,6)
    return Train_Loss
    
def validate(validation_loader, model, criterion, num_classes, device, activation, model_path=False, BINARY_SEG=None, THRESHOLD=None):
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    iou_calculator = IoU(num_classes)
    
    total_loss = 0.0
    total_samples = 0
    
    columns = ['image_path'] + [f'class_{i}' for i in range(num_classes)]
    metrics_df = pd.DataFrame(columns=['image_path'] + [f'{metric}_class_{i}' for metric in ['iou'] for i in range(num_classes)])

    for i, (input, target, image_path) in enumerate(tqdm(validation_loader, desc="Valiation", unit="batch", leave=False)):
        input = input.to(device)
        target = target.to(device)
        batch_size = input.size(0)
        with torch.no_grad():
            output = activation(model(input))
            loss = criterion(output, target).float()
            
        output_np = np.squeeze(np.where(output.cpu().numpy() > THRESHOLD, 1, 0), axis=1) if BINARY_SEG else torch.argmax(output, dim=1).cpu().numpy()
        target_np = target.cpu().numpy()
        target_np = np.squeeze(target_np, axis=1)
        
        presence_matrix = check_class_presence(target_np, num_classes=num_classes)
        
        iou_values = iou_calculator(output_np, target_np)
        iou_values *= presence_matrix

        metrics_row = {'image_path': image_path}
        for j in range(num_classes):
            if presence_matrix[:, j].any():
                metrics_row[f'iou_class_{j}'] = iou_values[:, j].mean() if presence_matrix[:, j].any() else np.nan
        # metrics_df = metrics_df.append(metrics_row, ignore_index=True)

        metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics_row])], ignore_index=True)
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    mean_loss = total_loss / total_samples
    return metrics_df, mean_loss 

def test(test_loader, model, criterion, device, num_classes, activation, model_path=False, BINARY_SEG=None, THRESHOLD=None, SAVE_RESULT=None, vis_root=None):
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    
    dice_calculator = DiceCoefficient(num_classes)
    iou_calculator = IoU(num_classes)
    precision_calculator = Precision(num_classes) 
    recall_calculator = Recall(num_classes) 
    
    total_loss = 0.0
    total_samples = 0
    if SAVE_RESULT:
        save_n = 0
        save_bool = False
    columns = ['image_path'] + [f'class_{i}' for i in range(num_classes)]
    metrics_df = pd.DataFrame(columns=['image_path'] + [f'{metric}_class_{i}' for metric in ['iou', 'dice', 'precision', 'recall'] for i in range(num_classes)])

    for i, (input, target, image_path) in enumerate(tqdm(test_loader, desc="Test", unit="batch", leave=False)):
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = activation(model(input))
            loss = criterion(output, target).float()

        output_np = np.squeeze(np.where(output.cpu().numpy() > THRESHOLD, 1, 0), axis=1) if BINARY_SEG else torch.argmax(output, dim=1).cpu().numpy()
        target_np = target.cpu().numpy()
        target_np = np.squeeze(target_np, axis=1)
        inputs_np = input.cpu().numpy().transpose(0, 2, 3, 1)  # (batch, H, W, C)


        presence_matrix = check_class_presence(target_np, num_classes=num_classes)

        # 메트릭 계산 (클래스가 존재하는 경우에만)
        iou_values = iou_calculator(output_np, target_np)
        dice_values = dice_calculator(output_np, target_np)
        precision_values = precision_calculator(output_np, target_np)
        recall_values = recall_calculator(output_np, target_np)
        
        # presence_matrix가 True인 경우에만 데이터프레임에 메트릭 값 추가
        batch_metrics = []
        for b in range(input.shape[0]):
            metrics_row = {'image_path': os.path.basename(image_path[b])}
            for j in range(num_classes):
                metrics_row[f'iou_class_{j}'] = iou_values[b, j] if presence_matrix[b, j] else np.nan
                metrics_row[f'dice_class_{j}'] = dice_values[b, j] if presence_matrix[b, j] else np.nan
                metrics_row[f'precision_class_{j}'] = precision_values[b, j] if presence_matrix[b, j] else np.nan
                metrics_row[f'recall_class_{j}'] = recall_values[b, j] if presence_matrix[b, j] else np.nan
            batch_metrics.append(metrics_row)

        metrics_df = pd.concat([metrics_df, pd.DataFrame(batch_metrics)], ignore_index=True)

        for b in range(input.shape[0]):
            fn = os.path.basename(image_path[b]) if isinstance(image_path, list) else os.path.basename(image_path)
            save_path = os.path.join(vis_root, f'{os.path.splitext(fn)[0]}_viz.png')
            save_segmentation_viz(
                img=inputs_np[b],
                gt_mask=target_np[b],
                pred_mask=output_np[b],
                save_path=save_path,
                binary=BINARY_SEG,
            )
        
        
        # 손실 업데이트
        total_loss += loss.item() * input.shape[0]
        total_samples += input.shape[0]

    mean_loss = total_loss / total_samples
    return metrics_df, mean_loss



def save_segmentation_viz(img, gt_mask, pred_mask, save_path, binary=True):
    """
    img: numpy array, shape (H, W, 3) or (H, W, 1)
    gt_mask/pred_mask: (H, W)
    save_path: str, destination filepath
    binary: True if binary mask, False for multi-class
    """
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[-1] == 3:
        img = img[..., [2, 1, 0]]
    img = np.clip((img * 255), 0, 255).astype(np.uint8)

    if binary:
        gt_show = np.stack([gt_mask * 255] * 3, axis=-1)
        pred_show = np.stack([pred_mask * 255] * 3, axis=-1)
    else:
        cmap = plt.get_cmap('jet')
        gt_show = (cmap(gt_mask / (gt_mask.max() if gt_mask.max() > 0 else 1))[:, :, :3] * 255).astype(np.uint8)
        pred_show = (cmap(pred_mask / (pred_mask.max() if pred_mask.max() > 0 else 1))[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    axes[0].imshow(img)
    axes[0].set_title('Input')
    axes[1].imshow(gt_show)
    axes[1].set_title('GT')
    axes[2].imshow(pred_show)
    axes[2].set_title('Pred')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    
def aggregate_measures(metrics_df, exclude_background=True, metric_types=['iou', 'dice', 'precision', 'recall']):
    # Calculate classwise_metrics: average across all image paths for each class and metric
    classwise_metrics = metrics_df.iloc[:, 1:].mean(skipna=True)  # Ignoring the 'image_path' column

    # Calculate samplewise_metrics: average across all classes for each sample (ignoring NaN values)
    # Grouping the metrics into IoU, Dice, Precision, and Recall
    samplewise_metrics_dict = {}

    for metric in metric_types:
        metric_columns = [col for col in metrics_df.columns if metric in col and (not exclude_background or not col.endswith('_0'))]
        samplewise_metrics_dict[f'mean_{metric}'] = metrics_df[metric_columns].mean(axis=1, skipna=True)

    # Create DataFrames for both results with the appropriate column names
    classwise_metrics_df = classwise_metrics.to_frame().T  # Convert to DataFrame with a single row

    # For samplewise_metrics, create a DataFrame with 'image_path' and all averaged metrics
    samplewise_metrics_df = metrics_df[['image_path']].copy()  # Keep 'image_path' column

    for key, value in samplewise_metrics_dict.items():
        samplewise_metrics_df[key] = value
    overall_metrics = {}

    for metric in metric_types:
        metric_columns = [col for col in classwise_metrics_df.columns if metric in col and (not exclude_background or not col.endswith('_0'))]
        overall_metrics[metric] = classwise_metrics_df[metric_columns].mean(axis=1, skipna=True).values[0]

    overall_metrics_df = pd.DataFrame(overall_metrics, index=[0])
    
    return overall_metrics_df, classwise_metrics_df, samplewise_metrics_df

class LossSaver(object):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    def reset(self):
        self.train_losses = []
        self.val_losses = []
    def update(self, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
    def return_list(self):
        return self.train_losses, self.val_losses
    def save_as_csv(self, csv_file):
        df = pd.DataFrame({'Train Losses': self.train_losses, 'Validation Losses': self.val_losses})
        df.index = [f"{i+1} Epoch" for i in df.index]
        df.to_csv(csv_file, index=True)
        
class AverageMeter (object):
    def __init__(self):
        self.reset ()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count        
def Execute_Experiment(model_name, model, Dataset_Name, train_loader, validation_loader, test_loader, 
                            optimizer, lr_scheduler, criterion, number_of_classes, df, epochs, device, output_dir,
                           BINARY_SEG, exclude_background, out_channels, seed, THRESHOLD, EARLY_STOP, SAVE_RESULT, vis_root, Experiments_Time
                           ):
    start = timeit.default_timer()
    activation = nn.Sigmoid() if BINARY_SEG else nn.Softmax(1)
    os.makedirs(output_dir, exist_ok = True)
    control_random_seed(seed)
    # Train
    now = datetime.now()
    Train_date=now.strftime("%y%m%d_%H%M%S")
    print('Training Start Time:',Train_date)
    model_path=output_dir+f'/{Train_date}_{model_name}_{Dataset_Name}.pt'
    best=9999
    best_epoch=1
    Early_Stop=0
    loss_saver = LossSaver()
    train_start_time = timeit.default_timer()
    for epoch in range(1, epochs+1):
        Train_Loss = train(train_loader, epoch, model, criterion, optimizer, device, activation)
        lr_scheduler.step()
        metrics_df, Val_Loss  = validate(validation_loader, model, criterion, number_of_classes, device, activation, BINARY_SEG=BINARY_SEG, THRESHOLD=THRESHOLD)
        
        overall_metrics_df, classwise_metrics_df, samplewise_metrics_df = aggregate_measures(metrics_df, exclude_background, metric_types = ['iou'])
        
        Val_IoU = overall_metrics_df['iou'].item(); 

        date = datetime.now().strftime("%y%m%d_%H%M%S")
        print(f"{epoch}EP({date}): T_Loss: {Train_Loss:.6f} V_Loss: {Val_Loss:.6f} IoU: {Val_IoU:.4f}", end=' ')
        
        loss_saver.update(Train_Loss, Val_Loss)
        loss_saver.save_as_csv(f'{output_dir}/Losses_{Train_date}.csv')
        if Val_Loss<best:
            Early_Stop = 0
            torch.save(model.state_dict(), model_path)
            best_epoch = epoch
            best = Val_Loss
            print(f'Best Epoch: {best_epoch} Loss: {Val_Loss:.6f}')
        else:
            print('')
            Early_Stop+=1
        if Early_Stop>=EARLY_STOP:
            break
    train_stop_time = timeit.default_timer()
    # test
    now = datetime.now()
    date=now.strftime("%y%m%d_%H%M%S")
    print('Test Start Time:',date)
    metrics_df, Test_Loss = test(test_loader, model, criterion, device, number_of_classes, activation, model_path=model_path, BINARY_SEG=BINARY_SEG, THRESHOLD=THRESHOLD, SAVE_RESULT=SAVE_RESULT, vis_root=vis_root)
    overall_metrics_df, classwise_metrics_df, samplewise_metrics_df = aggregate_measures(metrics_df, exclude_background, metric_types = ['iou', 'dice', 'precision', 'recall'])
    
    metrics_df.to_csv(f'{output_dir}/Test_sample_class_wise_{model_name}_{Dataset_Name}_{Train_date}.csv', index=False, header=True, encoding="cp949")
    
    classwise_metrics_df.to_csv(f'{output_dir}/Test_class_wise_{model_name}_Iter_{Dataset_Name}_{Train_date}.csv', index=False, header=True, encoding="cp949")
    samplewise_metrics_df.to_csv(f'{output_dir}/Test_sample_wise_{model_name}_{Dataset_Name}_{Train_date}.csv', index=False, header=True, encoding="cp949")
            
    iou = overall_metrics_df['iou'].item(); dice = overall_metrics_df['dice'].item(); 
    precision = overall_metrics_df['precision'].item(); recall = overall_metrics_df['recall'].item();

    date = datetime.now().strftime("%y%m%d_%H%M%S")
    print('Best Epoch:', best_epoch)

    print(f"Test({date}): Loss: {Test_Loss:.6f} IoU: {iou:.4f} Dice: {dice:.4f} Precision: {precision:.4f} Recall: {recall:.4f}")

    stop = timeit.default_timer();m, s = divmod((train_stop_time - train_start_time)/epoch, 60);h, m = divmod(m, 60);Time_per_Epoch = "%02d:%02d:%02d" % (h, m, s);
    m, s = divmod(stop - start, 60);h, m = divmod(m, 60);Time = "%02d:%02d:%02d" % (h, m, s);
    total_params = sum(p.numel() for p in model.parameters()); total_params = format(total_params , ',');

    Performances = [Experiments_Time, Train_date, Dataset_Name, 
                    model_name, best, Test_Loss, iou, dice, precision, recall, total_params, Time]
    new_row = pd.DataFrame([Performances], columns=df.columns)
    df = pd.concat([df, new_row], ignore_index=True)
    now = datetime.now()
    date=now.strftime("%y%m%d_%H%M%S")
    print('End',date)
    
    return df

def extend_to_full_batch(image_path_list, target_path_list, batch_size):
    # 원본 리스트의 길이를 batch_size로 나누어떨어지게 늘리는 함수
    num_samples = len(image_path_list)
    remainder = num_samples % batch_size

    if remainder != 0:
        # 필요한 만큼을 무작위로 추가해서 채운다
        extra_needed = batch_size - remainder
        available_indices = list(range(num_samples))  # 모든 인덱스에서 무작위로 추가할 인덱스 선택
        random_indices = random.choices(available_indices, k=extra_needed)  # 필요한 개수만큼 무작위로 선택

        for i in random_indices:
            image_path_list.append(image_path_list[i])
            target_path_list.append(target_path_list[i])
    return image_path_list,target_path_list