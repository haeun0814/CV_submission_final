import torch    
import torch.nn.functional as F
def Make_Optimizer(model):
    # TODO: 모델의 파라미터와 원하는 하이퍼파라미터(learning rate 등)를 사용하여
    #       optimizer(예: torch.optim.Adam, torch.optim.SGD 등)를 생성 및 반환하도록 작성하세요.
    return optimizer
def Make_LR_Scheduler(optimizer):
    # TODO: optimizer와 원하는 학습률 스케줄링 전략(예: CosineAnnealingLR, StepLR 등)을 사용하여
    #       lr_scheduler를 생성 및 반환하도록 작성하세요.
    return lr_scheduler

def Make_Loss_Function(number_of_classes):
    # TODO: 클래스 수에 따라 적절한 loss function(예: CrossEntropyLoss 등)을
    #       생성 및 반환하도록 작성하세요.
    return loss_function
    

# 예시용 (제출시 삭제)
def Make_Optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def Make_LR_Scheduler(optimizer):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1e-6, eta_min = 30)

def Make_Loss_Function(number_of_classes):
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
    
    BINARY_SEG = True if number_of_classes==2 else False
    return DiceCELoss(mode='binary') if BINARY_SEG else DiceCELoss(mode='multiclass') 