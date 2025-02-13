import torch
from torch import nn

class UncertaintyLoss(nn.Module):
    def __init__(self, num_tasks=2):
        super().__init__()
        # Initialize learnable log variance parameters for each task
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        # Iterate over all tasks to compute the weighted loss
        total_loss = 0.0
        for i, loss in enumerate(losses):
            # Each task's loss is weighted by exp(-log_var) and a penalty term is added
            total_loss += torch.exp(-self.log_vars[i]) * loss + self.log_vars[i]
        return total_loss

def compute_mtl_loss(outputs, y_reg, y_cls, regression_criterion, classification_criterion, uncertainty_loss=UncertaintyLoss(num_tasks=2)):
    """
    Compute multi-task learning loss with uncertainty weighting.
    Note: `uncertainty_loss` is passed as an argument and should be instantiated once
    (e.g., as part of your model) so that its parameters persist and are updated during training.
    """
    # Compute individual task losses
    regression_loss = regression_criterion(outputs['regression_output'], y_reg)
    classification_loss = classification_criterion(outputs['classification_output'], y_cls)
    
    # Compute the overall loss using uncertainty weighting
    total_loss = uncertainty_loss([regression_loss, classification_loss])
    
    return total_loss, regression_loss, classification_loss
