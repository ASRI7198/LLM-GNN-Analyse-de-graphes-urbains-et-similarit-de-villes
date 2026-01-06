import sys
sys.path.append('C:/Users/DELL/Desktop/LLM-GNN')

import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import negative_sampling
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
import config as args

def train(model, train_data, val_data):
    train_losses = []
    val_losses = []
    val_aucc = []
    val_accc = []

    optimizer = optim.Adam(model.parameters(), lr=args.LR)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(1, args.EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        z = model.encode(train_data.x, train_data.edge_index)

        neg_edge_index = negative_sampling(
            edge_index=train_data.edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_data.edge_label_index.size(1),
            method='sparse'
        )


        edge_label_index = torch.cat([train_data.edge_label_index, neg_edge_index], dim=-1)
        edge_label = torch.cat([
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1))
        ], dim=0)

        out = model.decode(z, edge_label_index).view(-1)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            z_val = model.encode(val_data.x, val_data.edge_index)

            pos_pred = model.decode(z_val, val_data.edge_label_index).view(-1)
            pos_label = val_data.edge_label

            neg_edge_index_val = negative_sampling(
                edge_index=val_data.edge_index,
                num_nodes=val_data.num_nodes,
                num_neg_samples=val_data.edge_label_index.size(1),
                method='sparse'
            )
            neg_pred = model.decode(z_val, neg_edge_index_val).view(-1)
            neg_label = torch.zeros(neg_edge_index_val.size(1), device=z_val.device)

            y_pred = torch.cat([pos_pred, neg_pred], dim=0)
            y_true = torch.cat([pos_label, neg_label], dim=0)

            val_pred_sigmoid = torch.sigmoid(y_pred)
            val_auc = roc_auc_score(y_true.cpu().numpy(), val_pred_sigmoid.cpu().numpy())
            val_acc = accuracy_score(y_true.cpu().numpy(), (val_pred_sigmoid.cpu().numpy() > 0.5).astype(int))
            val_loss = criterion(y_pred, y_true).item()

        val_losses.append(val_loss)
        val_aucc.append(val_auc)
        val_accc.append(val_acc)

        if val_loss < best_val_loss - args.MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.3f}, Val ACC: {val_acc:.3f}")

        if epochs_no_improve >= args.PATIENCE:
            print(f"⏱️ Early stopping : plus d’amélioration depuis {args.PATIENCE} époques.")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

 
    epochs_list = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, label='Train Loss')
    plt.plot(epochs_list, val_losses,  label='Val Loss')
    plt.plot(epochs_list, val_aucc,   label='Val AUC')
    plt.plot(epochs_list, val_accc,   label='Val ACC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.title('Training Metrics (Loss, AUC, ACC)')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.VIS_LOSS_TRAIN_VAL_GAT, bbox_inches='tight')
    #plt.show()

    return model
