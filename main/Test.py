import torch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, accuracy_score


def test(model, test_data):
    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        z = model.encode(test_data.x, test_data.edge_index)

        pos_pred = model.decode(z, test_data.edge_label_index).view(-1)
        pos_label = test_data.edge_label

        neg_edge_index = negative_sampling(
            edge_index=test_data.edge_index,
            num_nodes=test_data.num_nodes,
            num_neg_samples=test_data.edge_label_index.size(1),
            method='sparse'
        )
        neg_pred = model.decode(z, neg_edge_index).view(-1)
        neg_label = torch.zeros(neg_edge_index.size(1), device=z.device)

        y_pred = torch.cat([pos_pred, neg_pred], dim=0)
        y_true = torch.cat([pos_label, neg_label], dim=0)

        y_prob = torch.sigmoid(y_pred)

        test_loss = criterion(y_pred, y_true).item()
        test_auc = roc_auc_score(y_true.cpu().numpy(), y_prob.cpu().numpy())
        test_acc = accuracy_score(
            y_true.cpu().numpy(),
            (y_prob.cpu().numpy() > 0.5).astype(int)
        )

    return test_loss, test_auc, test_acc
