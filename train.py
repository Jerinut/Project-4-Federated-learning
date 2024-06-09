# # import torch
# # import torch.optim as optim
# # from model import create_model
# # from data import load_datasetset
# # from collections import OrderedDict
# # from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
# # import numpy as np
# # import torch.nn.functional as F
# # import os
# # fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
# # client_id= os.environ.get('client_id')
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # class NeuralNetworkManager:
# #     @staticmethod
# #     def get_parameters(net) -> list:
# #         return [val.cpu().numpy() for _, val in net.state_dict().items()]

# #     @staticmethod
# #     def set_parameters(net, parameters: list):
# #         params_dict = zip(net.state_dict().keys(), parameters)
# #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if len(v.shape) > 0})
# #         net.load_state_dict(state_dict, strict=False)
# #         return net

# #     @staticmethod
# #     def train(net, trainloader, epochs: int):
# #         criterion = torch.nn.CrossEntropyLoss()
# #         optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# #         net.train()
# #         for epoch in range(epochs):
# #             correct, total, epoch_loss = 0, 0, 0.0
# #             for batch in trainloader:
# #                 images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
# #                 optimizer.zero_grad()
# #                 outputs = net(images)
# #                 loss = criterion(outputs, labels)
# #                 loss.backward()
# #                 optimizer.step()
# #                 epoch_loss += loss.item()
# #                 total += labels.size(0)
# #                 correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
# #             print(f"Epoch {epoch+1}: train loss {epoch_loss / len(trainloader)}, accuracy {correct / total}")

# #     @staticmethod
# #     def test(net, testloader):
# #         criterion = torch.nn.CrossEntropyLoss()
# #         correct, total, loss = 0, 0, 0.0
# #         net.eval()
# #         all_preds, all_labels, all_probs = [], [], []
# #         with torch.no_grad():
# #             for batch in testloader:
# #                 images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
# #                 outputs = net(images)
# #                 loss += criterion(outputs, labels).item()
# #                 _, predicted = torch.max(outputs.data, 1)
# #                 all_preds.extend(predicted.cpu().numpy())
# #                 all_labels.extend(labels.cpu().numpy())
# #                 all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
# #                 total += labels.size(0)
# #                 correct += (predicted == labels).sum().item()
# #         metrics = {
# #             "accuracy": accuracy_score(all_labels, all_preds),
# #             "f1": f1_score(all_labels, all_preds, average='weighted'),
# #             "kappa": cohen_kappa_score(all_labels, all_preds),
# #             "roc_auc": roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else np.nan
# #         }
# #         return loss / len(testloader.dataset), metrics

# # def main():
# #     # fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
# #     # client_id= os.environ.get('client_id')
# #     trainloader, testloader = load_datasetset(fedn_data_path, client_id)
# #     net = create_model().to(DEVICE)
# #     NeuralNetworkManager.train(net, trainloader, epochs=4)
# #     loss, metrics = NeuralNetworkManager.test(net, testloader)
# #     print(f"Test loss: {loss}, Metrics: {metrics}")

# # if __name__ == "__main__":
# #     main()
# # ###############################################Second Itration ###############################################
# ###############################################################################################################
# # import os
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # from model import VGG, compile_model
# # from data import load_datasetset, FEDN_DATA_PATH
# # from collections import OrderedDict
# # from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score

# # DEVICE = torch.device("cpu")

# # class NeuralNetworkManager:
# #     @staticmethod
# #     def get_parameters(net):
# #         return [val.cpu().numpy() for _, val in net.state_dict().items()]

# #     @staticmethod
# #     def set_parameters(net, parameters):
# #         params_dict = zip(net.state_dict().keys(), parameters)
# #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
# #         net.load_state_dict(state_dict, strict=True)
# #         return net

# #     @staticmethod
# #     def train(net, trainloader, epochs=1):
# #         criterion = nn.CrossEntropyLoss()
# #         optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# #         net.train()
# #         for _ in range(epochs):
# #             for batch in trainloader:
# #                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
# #                 optimizer.zero_grad()
# #                 outputs = net(images)
# #                 loss = criterion(outputs, labels)
# #                 loss.backward()
# #                 optimizer.step()

# #     @staticmethod
# #     def test(net, testloader):
# #         criterion = nn.CrossEntropyLoss()
# #         correct, total, loss = 0, 0, 0.0
# #         net.eval()
# #         all_preds = []
# #         all_labels = []
# #         all_probs = []
# #         with torch.no_grad():
# #             for batch in testloader:
# #                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
# #                 outputs = net(images)
# #                 loss += criterion(outputs, labels).item()
# #                 _, predicted = torch.max(outputs.data, 1)
# #                 all_preds.extend(predicted.cpu().numpy())
# #                 all_labels.extend(labels.cpu().numpy())
# #                 all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
# #                 total += labels.size(0)
# #                 correct += (predicted == labels).sum().item()

# #         loss /= len(testloader)
# #         accuracy = correct / total
# #         f1 = f1_score(all_labels, all_preds, average='weighted')
# #         kappa = cohen_kappa_score(all_labels, all_preds)
# #         roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else float('nan')
        
# #         metrics = {"accuracy": accuracy, "f1": f1, "kappa": kappa, "roc_auc": roc_auc}
# #         return loss, metrics

# # def main():
# #     client_id = 0  # Replace with actual client ID as needed
# #     model = compile_model().to(DEVICE)
# #     trainloader, testloader = load_datasetset()
# #     NeuralNetworkManager.train(model, trainloader, epochs=4)
# #     loss, metrics = NeuralNetworkManager.test(model, testloader)
# #     print(f"Validation Loss: {loss}")
# #     print(f"Metrics: {metrics}")

# # if __name__ == "__main__":
# #     main()
# #########################################################Third Itration ############################################
# #####################################################################################################################
# import os
# import sys
# import torch
# import torch.optim as optim
# from model import create_model, load_parameters, save_parameters
# from data import load_datasetset
# from collections import OrderedDict
# from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
# import numpy as np
# import torch.nn.functional as F
# from fedn.utils.helpers.helpers import save_metrics, save_metadata

# # Fetch environment variables
# fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
# client_id = os.environ.get('client_id', 'default_client_id')
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class NeuralNetworkManager:
#     @staticmethod
#     def get_parameters(net):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     @staticmethod
#     def set_parameters(net, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)
#         return net

#     @staticmethod
#     def train(net, trainloader, epochs=1, lr=0.01):
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
#         net.train()
#         for epoch in range(epochs):
#             epoch_loss = 0.0
#             correct, total = 0, 0
#             for batch in trainloader:
#                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
#                 optimizer.zero_grad()
#                 outputs = net(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#                 epoch_loss += loss.item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#             print(f"Epoch {epoch + 1}: Train Loss: {epoch_loss / len(trainloader)}, Accuracy: {correct / total}")

#     @staticmethod
#     def test(net, testloader):
#         criterion = torch.nn.CrossEntropyLoss()
#         correct, total, loss = 0, 0, 0.0
#         net.eval()
#         all_preds, all_labels, all_probs = [], [], []
#         with torch.no_grad():
#             for batch in testloader:
#                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
#                 outputs = net(images)
#                 loss += criterion(outputs, labels).item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         loss /= len(testloader)
#         accuracy = correct / total
#         f1 = f1_score(all_labels, all_preds, average='weighted')
#         kappa = cohen_kappa_score(all_labels, all_preds)
#         roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else float('nan')
        
#         metrics = {"accuracy": accuracy, "f1": f1, "kappa": kappa, "roc_auc": roc_auc}
#         return loss, metrics

# def main():
#     # Load dataset
#     trainloader, testloader = load_datasetset(fedn_data_path, client_id)
    
#     # Initialize model
#     net = create_model().to(DEVICE)
    
#     # Load initial parameters
#     in_model_path = sys.argv[1]
#     out_model_path = sys.argv[2]
#     net = load_parameters(in_model_path)
    
#     # Train model
#     NeuralNetworkManager.train(net, trainloader, epochs=4, lr=0.01)
    
#     # Test model
#     loss, metrics = NeuralNetworkManager.test(net, testloader)
#     print(f"Validation Loss: {loss}")
#     print(f"Metrics: {metrics}")
    
#     # Save model parameters
#     save_parameters(net, out_model_path)
    
#     # Save metadata
#     metadata = {
#         "num_examples": len(trainloader.dataset),
#         "batch_size": trainloader.batch_size,
#         "epochs": 4,
#         "lr": 0.01,
#     }
#     save_metadata(metadata, out_model_path)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python train.py <in_model_path> <out_model_path>")
#         sys.exit(1)
#     main()
# import os
# import sys
# import torch
# import math
# import torch.nn as nn
# import torch.optim as optim
# from model import VGG, compile_model, load_parameters, save_parameters
# from data import load_dataset, FEDN_DATA_PATH
# from collections import OrderedDict
# from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
# from fedn.utils.helpers.helpers import save_metadata

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class NeuralNetworkManager:
#     @staticmethod
#     def get_parameters(net):
#         return [val.cpu().numpy() for _, val in net.state_dict().items()]

#     @staticmethod
#     def set_parameters(net, parameters):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#         net.load_state_dict(state_dict, strict=True)
#         return net

#     @staticmethod
#     def train(net, trainloader, epochs=1):
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#         net.train()
#         for _ in range(epochs):
#             for batch in trainloader:
#                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
#                 optimizer.zero_grad()
#                 outputs = net(images)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#     @staticmethod
#     def test(net, testloader):
#         criterion = nn.CrossEntropyLoss()
#         correct, total, loss = 0, 0, 0.0
#         net.eval()
#         all_preds = []
#         all_labels = []
#         all_probs = []
#         with torch.no_grad():
#             for batch in testloader:
#                 images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
#                 outputs = net(images)
#                 loss += criterion(outputs, labels).item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         loss /= len(testloader)
#         accuracy = correct / total
#         f1 = f1_score(all_labels, all_preds, average='weighted')
#         kappa = cohen_kappa_score(all_labels, all_preds)
#         roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else float('nan')
        
#         metrics = {"accuracy": accuracy, "f1": f1, "kappa": kappa, "roc_auc": roc_auc}
#         return loss, metrics

# def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=4, lr=0.01):
#     """ Complete a model update.

#     Load model paramters from in_model_path (managed by the FEDn client),
#     perform a model update, and write updated paramters
#     to out_model_path (picked up by the FEDn client).

#     :param in_model_path: The path to the input model.
#     :type in_model_path: str
#     :param out_model_path: The path to save the output model to.
#     :type out_model_path: str
#     :param data_path: The path to the data file.
#     :type data_path: str
#     :param batch_size: The batch size to use.
#     :type batch_size: int
#     :param epochs: The number of epochs to train.
#     :type epochs: int
#     :param lr: The learning rate to use.
#     :type lr: float
#     """
#     # Use environment variable for data path if not provided
#     if data_path is None:
#         data_path = FEDN_DATA_PATH
#         if data_path is None:
#             raise ValueError("Data path must be provided or set in the environment variable 'FEDN_DATA_PATH'.")

#     # Load data
#     x_train, y_train = load_dataset(data_path)

#     # Load parameters and initialize model
#     model = load_parameters(in_model_path).to(DEVICE)

#     # Train
#     optimizer = torch.optim.SGD(model.parameters(), lr=lr)
#     n_batches = int(math.ceil(len(x_train) / batch_size))
#     criterion = nn.CrossEntropyLoss()
#     for e in range(epochs):  # epoch loop
#         for b in range(n_batches):  # batch loop
#             # Retrieve current batch
#             batch_x = x_train[b * batch_size:(b + 1) * batch_size]
#             batch_y = y_train[b * batch_size:(b + 1) * batch_size]
#             # Train on batch
#             optimizer.zero_grad()
#             outputs = model(batch_x.to(DEVICE))
#             loss = criterion(outputs, batch_y.to(DEVICE))
#             loss.backward()
#             optimizer.step()
#             # Log
#             if b % 100 == 0:
#                 print(f"Epoch {e}/{epochs-1} | Batch: {b}/{n_batches-1} | Loss: {loss.item()}")

#     # Metadata needed for aggregation server side
#     metadata = {
#         # num_examples are mandatory
#         'num_examples': len(x_train),
#         'batch_size': batch_size,
#         'epochs': epochs,
#         'lr': lr
#     }
#     metadata_path = out_model_path + ".metadata"
#     # Save JSON metadata file (mandatory)
#     save_metadata(metadata, metadata_path)

#     # Save model update (mandatory)
#     save_parameters(model, out_model_path)

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python train.py <in_model_path> <out_model_path>")
#         sys.exit(1)
#     train(sys.argv[1], sys.argv[2])
import os
import sys
import torch
import math
import torch.nn as nn
import torch.optim as optim
from model import compile_model, load_parameters, save_parameters
from data import load_dataset, FEDN_DATA_PATH, client_id
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
from fedn.utils.helpers.helpers import save_metadata

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetworkManager:
    @staticmethod
    def get_parameters(net):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    @staticmethod
    def set_parameters(net, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        return net

    @staticmethod
    def train(net, trainloader, epochs=1, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        net.train()
        for epoch in range(epochs):
            for i, batch in enumerate(trainloader):
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if i % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {loss.item():.4f}")

    @staticmethod
    def test(net, testloader):
        criterion = nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.nn.functional.softmax(outputs, dim=1).cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss /= len(testloader)
        accuracy = correct / total
        f1 = f1_score(all_labels, all_preds, average='weighted')
        kappa = cohen_kappa_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else float('nan')
        
        metrics = {"accuracy": accuracy, "f1": f1, "kappa": kappa, "roc_auc": roc_auc}
        return loss, metrics

def train(in_model_path, out_model_path, data_path=None, batch_size=32, epochs=4, lr=0.01):
    """ Complete a model update.

    Load model parameters from in_model_path (managed by the FEDN client),
    perform a model update, and write updated parameters
    to out_model_path (picked up by the FEDN client).

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_model_path: The path to save the output model to.
    :type out_model_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    :param batch_size: The batch size to use.
    :type batch_size: int
    :param epochs: The number of epochs to train.
    :type epochs: int
    :param lr: The learning rate to use.
    :type lr: float
    """
    # Use environment variable for data path if not provided
    if data_path is None:
        data_path = FEDN_DATA_PATH
        if data_path is None:
            raise ValueError("Data path must be provided or set in the environment variable 'FEDN_DATA_PATH'.")

    # Load data
    trainloader, testloader = load_dataset()

    if trainloader is None or testloader is None:
        raise ValueError(f"Data loading failed. Check the data directory structure and files for client {client_id}.")

    # Load parameters and initialize model
    model = load_parameters(in_model_path).to(DEVICE)
    
    # Train the model
    NeuralNetworkManager.train(model, trainloader, epochs=epochs, lr=lr)

    # Metadata needed for aggregation server side
    metadata = {
        'num_examples': len(trainloader.dataset),
        'batch_size': batch_size,
        'epochs': epochs,
        'lr': lr
    }
    metadata_path = out_model_path + ".metadata"

    # Debugging: Print metadata to ensure correctness
    print(f"Saving metadata to: {metadata_path}")
    print(f"Metadata content: {metadata}")

    # Save JSON metadata file (mandatory)
    try:
        save_metadata(metadata, metadata_path)
        print(f"Metadata saved successfully to {metadata_path}")
    except Exception as e:
        print(f"Failed to save metadata: {str(e)}")

    # Save model update (mandatory)
    save_parameters(model, out_model_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python train.py <in_model_path> <out_model_path>")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2])
