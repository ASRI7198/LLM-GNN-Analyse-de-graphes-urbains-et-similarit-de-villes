
import torch
from sklearn.metrics import roc_auc_score,confusion_matrix,roc_curve,average_precision_score


@torch.no_grad()
def eval_link_predictor(model, data):

    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    AUC = roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    ACC = average_precision_score(data.edge_label.cpu().numpy(), out.cpu().numpy())
    
    return AUC , ACC