import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from monai.losses import TverskyLoss

 
def Make_Optimizer(model):
    return  torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

def Make_LR_Scheduler(optimizer):
    return WarmupCosineLR(
            optimizer,
            T_max=50,               
            warmup_iters=5,       
            warmup_factor=1/10,     
            eta_min=1e-6) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1e-6, eta_min = 30)

def Make_Loss_Function(number_of_classes):
    
    BINARY_SEG = True if number_of_classes==2 else False
    WORK_MODE = "binary" if BINARY_SEG else "multiclass"
    
    if BINARY_SEG:
        loss = DiceCELoss(weight=0.5, mode='binary')
    else:
        loss = UniformCBCE_LovaszProb(number_of_classes)
    
    return loss

class WarmupCosineLR(_LRScheduler):
    """
    Cosine annealing + linear warm-up (epoch 단위 스케줄)
    """
    def __init__(
        self,
        optimizer,
        T_max: int,
        cur_iter: int = 0,              # 외부 호환용. epoch 기준이면 0, iter 기준이면 현재 step 수
        warmup_factor: float = 1.0 / 3,
        warmup_iters: int = 500,
        eta_min: float = 0.0,
    ):
        self.warmup_factor = warmup_factor
        self.warmup_iters  = warmup_iters
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=cur_iter - 1)

    def get_lr(self):
        # 첫 호출( last_epoch == -1 ) → base_lr 그대로
        if self.last_epoch == -1:
            return self.base_lrs

        # 1) Warm-up 구간
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / float(max(1, self.warmup_iters))
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]

        # 2) Cosine annealing 구간
        progress = (self.last_epoch - self.warmup_iters) / float(
            max(1, self.T_max - self.warmup_iters)
        )
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress)) / 2
            for base_lr in self.base_lrs
        ]

class DiceCELoss:
    def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
        self.weight = weight
        self.epsilon = epsilon
        self.mode = mode
    
    def __call__(self, pred, target):
        if self.mode == 'binary':
            pred = pred.squeeze(1)  # shape: (batchsize, H, W)
            target = target.squeeze(1).float()
            intersection = torch.sum(pred * target, dim=(1, 2))
            union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.binary_cross_entropy(pred, target)
        
        elif self.mode == 'multiclass':
            batchsize, num_classes, H, W = pred.shape
            target = target.squeeze(1)
            target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
            intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
            union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
            dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
            dice_loss = 1 - dice.mean()
            
            ce_loss = F.cross_entropy(pred, target)
        else:
            raise ValueError("mode should be 'binary' or 'multiclass'")
        
        combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
        
        return combined_loss



class UniformCBCE_LovaszProb(nn.Module):
    """
    · 입력  : Softmax 확률맵  (B, C, H, W)
    · 출력  : CE(NLL) + Lovász-Softmax 결합 손실
    · 배경(class 0) 은 bg_factor 로 가중치 축소
    """
    def __init__(
        self,
        num_classes: int,
        bg_factor: float = 0.05,
        coef_ce: float = 0.6,
        coef_iou: float = 0.4,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps
        # --- 클래스별 가중치 (배경 축소) ---
        weight = torch.ones(num_classes, dtype=torch.float32)
        weight[0] = bg_factor
        self.register_buffer("ce_weight", weight)  # → 단순 텐서로만 보관

        self.ce_w = coef_ce
        self.iou_w = coef_iou
        self.num_classes = num_classes

    # -------------------------------------------------
    # 1) 결정론 Cross-Entropy (GPU에서도 deterministic)
    # -------------------------------------------------
    @staticmethod
    def ce2d_deterministic(logits, target, weight=None, ignore_index=-100, eps=1e-7):
        """
        logits : (N, C, H, W)  – Softmax 적용 전 값
        target : (N, H, W)     – 정수 라벨
        weight : (C,) or None  – 클래스 가중치
        """
        log_p = torch.log_softmax(logits, dim=1)                # (N,C,H,W)

        # gather 로 GT 위치의 log-prob만 추출
        log_p = log_p.gather(1, target.unsqueeze(1))            # (N,1,H,W) – det.
        log_p = log_p.squeeze(1)                                # (N,H,W)

        mask = (target != ignore_index)
        if weight is not None:
            w = weight[target]                                  # 클래스별 가중치 맵
            loss = -(w * log_p * mask).sum() / (w * mask).sum()
        else:
            loss = -(log_p * mask).sum() / mask.sum()

        return loss

    # -------------------------------------------------
    # 2) Lovász-Softmax (확률 입력, deterministic OK)
    # -------------------------------------------------
    @staticmethod
    def lovasz_softmax(probs, labels, eps=1e-6):
        """
        probs  : (B,C,H,W) – Softmax 확률
        labels : (B,H,W)   – 정수 라벨
        """
        B, C, H, W = probs.shape
        losses = []
        for c in range(C):
            fg = (labels == c).float()                     # (B,H,W)
            if fg.sum() == 0:
                continue
            pc = probs[:, c]                               # (B,H,W)
            errors = (fg - pc).abs()

            errors_sorted, perm = torch.sort(
                errors.view(B, -1), dim=1, descending=True
            )
            fg_sorted = fg.view(B, -1).gather(1, perm)

            grad = torch.cumsum(fg_sorted, dim=1)
            grad = grad / fg_sorted.sum(dim=1, keepdim=True).clamp(min=1)

            loss = (errors_sorted * grad).sum(dim=1) / (H * W)  # (B,)
            losses.append(loss)

        return torch.stack(losses).mean() if losses else torch.tensor(0., device=probs.device)

    # -------------------------------------------------
    def forward(self, probs, target):
        """
        probs  : Softmax 확률 (B,C,H,W)
        target : GT 라벨 (B,1,H,W) or (B,H,W)
        """
        if probs.dtype != torch.float32:
            probs = probs.float()

        if target.ndim == 4:            # (B,1,H,W) → (B,H,W)
            target = target.squeeze(1).long()
        else:
            target = target.long()

        # 1) CE 손실 (deterministic)
        loss_ce = self.ce2d_deterministic(
            logits=torch.log(probs.clamp(min=self.eps)),
            target=target,
            weight=self.ce_weight.to(probs.device),
        )

        # 2) Lovász-Softmax 손실
        loss_iou = self.lovasz_softmax(probs, target, eps=self.eps)

        return self.ce_w * loss_ce + self.iou_w * loss_iou


# def Make_LR_Scheduler(optimizer):
#     return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1e-6, eta_min = 30)

# def Make_Loss_Function(number_of_classes):
#     class DiceCELoss:
#         def __init__(self, weight=0.5, epsilon=1e-6, mode='multiclass'):
#             self.weight = weight
#             self.epsilon = epsilon
#             self.mode = mode
        
#         def __call__(self, pred, target):
#             if self.mode == 'binary':
#                 pred = pred.squeeze(1)  # shape: (batchsize, H, W)
#                 target = target.squeeze(1).float()
#                 intersection = torch.sum(pred * target, dim=(1, 2))
#                 union = torch.sum(pred, dim=(1, 2)) + torch.sum(target, dim=(1, 2))
#                 dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
#                 dice_loss = 1 - dice.mean()
                
#                 ce_loss = F.binary_cross_entropy(pred, target)
            
#             elif self.mode == 'multiclass':
#                 batchsize, num_classes, H, W = pred.shape
#                 target = target.squeeze(1)
#                 target_one_hot = F.one_hot(target, num_classes=num_classes).squeeze(1).permute(0, 3, 1, 2).float()
#                 intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
#                 union = torch.sum(pred, dim=(2, 3)) + torch.sum(target_one_hot, dim=(2, 3))
#                 dice = (2 * intersection + self.epsilon) / (union + self.epsilon)
#                 dice_loss = 1 - dice.mean()
                
#                 ce_loss = F.cross_entropy(pred, target)
#             else:
#                 raise ValueError("mode should be 'binary' or 'multiclass'")
            
#             combined_loss = self.weight * dice_loss + (1 - self.weight) * ce_loss
            
#             return combined_loss
    
#     BINARY_SEG = True if number_of_classes==2 else False
#     return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass') 