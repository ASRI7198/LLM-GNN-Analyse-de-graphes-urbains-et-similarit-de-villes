import sys
sys.path.append('C:/Users/DELL/Desktop/LLM-GNN')

import torch
import config as args
import torch_geometric.transforms as T
import Architectures.GCN as GCNModel
import Architectures.GAT as GATModel
import Architectures.Encoder as EncoderModel
import main.Train as TrainModule
import main.Test as TestModule
from torch_geometric.nn import VGAE


def split(data):
    s = T.RandomLinkSplit(
        num_val=0.05,
        num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=0.1,
    )
    train_data, val_data, test_data = s(data)
    return train_data, val_data, test_data

def operation(op):
    match op:
        case 1:
            model = GCNModel.GCN(
                in_channels=args.IN_CHANNELS,
                hidden_channels=args.HIDDEN_CHANNELS,
                hidden_channels2=args.HIDDEN_CHANNELS2,
                hidden_channels3=args.HIDDEN_CHANNELS3,
                out_channels=args.OUT_CHANNELS)
            return model
        case 2:
            model = GATModel.GAT(
                in_channels=args.IN_CHANNELS,
                hidden_channels=args.HIDDEN_CHANNELS,
                hidden_channels2=args.HIDDEN_CHANNELS2,
                hidden_channels3=args.HIDDEN_CHANNELS3,
                out_channels=args.OUT_CHANNELS,
                heads=args.HEADS)
            return model
        case 3:
            encoder = EncoderModel.Encoder(
                in_channels=args.IN_CHANNELS,
                hidden_channels=args.HIDDEN_CHANNELS,
                out_channels=args.OUT_CHANNELS)
            model = VGAE(encoder)
            return model
        case _:
            return "Opération inconnue"


if __name__ == "__main__":

    torch.cuda.empty_cache()
    data = torch.load(args.PATH_DATA_SAVE, weights_only=False)
    train_data, val_data, test_data = split(data)

    print("Données d'entraînement chargées avec succès.")

    choice = input("Veuillez indiquer votre choix : 1 pour GCN, 2 pour GAT, 3 pour VGAE : ")
    model = operation(int(choice))
    
    print("Modèle GAT initialisé avec succès.") 

    module = TrainModule.train(model,train_data,val_data)

    test_loss, test_auc, test_acc = TestModule.test(module, test_data)
    print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}, Test ACC: {test_acc:.4f}")

    torch.save(module, args.PATH_MODULE_SAVE_VGAE)
    print("Modèle entraîné et sauvegardé avec succès.")



    
    
    
    

    

    





    