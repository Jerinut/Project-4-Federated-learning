# # import torch
# # from model import create_model
# # from data import load_dataset
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
# #     def set_parameters(net, parameters: list):
# #         params_dict = zip(net.state_dict().keys(), parameters)
# #         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if len(v.shape) > 0})
# #         net.load_state_dict(state_dict, strict=False)
# #         return net

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

# # def validate_model():
# #     #data_path = "/var/data/"
# #     _, testloader = load_dataset(fedn_data_path)
# #     net = create_model().to(DEVICE)
# #     loss, metrics = NeuralNetworkManager.test(net, testloader)
# #     print(f"Validation loss: {loss}, Metrics: {metrics}")

# # if __name__ == "__main__":
# #     #client_id = os.getenv("CLIENT_ID", "0")
# #     validate_model()
# ########################################### This is the second itration #######################
# ###############################################################################################
# import os
# import sys
# import torch
# from model import create_model, load_parameters
# from data import load_dataset
# from fedn.utils.helpers.helpers import save_metrics
# from collections import OrderedDict
# from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
# import numpy as np
# import torch.nn.functional as F

# # Fetch environment variables
# fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
# client_id = os.environ.get('CLIENT_ID', 'default_client_id')
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class NeuralNetworkManager:
#     @staticmethod
#     def set_parameters(net, parameters: list):
#         params_dict = zip(net.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if len(v.shape) > 0})
#         net.load_state_dict(state_dict, strict=False)
#         return net

#     @staticmethod
#     def test(net, testloader):
#         criterion = torch.nn.CrossEntropyLoss()
#         correct, total, loss = 0, 0, 0.0
#         net.eval()
#         all_preds, all_labels, all_probs = [], [], []
#         with torch.no_grad():
#             for batch in testloader:
#                 images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
#                 outputs = net(images)
#                 loss += criterion(outputs, labels).item()
#                 _, predicted = torch.max(outputs.data, 1)
#                 all_preds.extend(predicted.cpu().numpy())
#                 all_labels.extend(labels.cpu().numpy())
#                 all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         metrics = {
#             "accuracy": accuracy_score(all_labels, all_preds),
#             "f1": f1_score(all_labels, all_preds, average='weighted'),
#             "kappa": cohen_kappa_score(all_labels, all_preds),
#             "roc_auc": roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else np.nan
#         }
#         return loss / len(testloader.dataset), metrics

# def validate(in_model_path, out_json_path, data_path=None):
#     """Validate model.

#     :param in_model_path: The path to the input model.
#     :type in_model_path: str
#     :param out_json_path: The path to save the output JSON to.
#     :type out_json_path: str
#     :param data_path: The path to the data file.
#     :type data_path: str
#     """
#     # Use environment variable for data path if not provided
#     if data_path is None:
#         data_path = fedn_data_path
#         if data_path is None:
#             raise ValueError("Data path must be provided or set in the environment variable 'FEDN_DATA_PATH'.")

#     # Load data
#     _, testloader = load_dataset(data_path)

#     # Load model
#     #net = create_model().to(DEVICE)
#     net = load_parameters(in_model_path)

#     # Evaluate
#     loss, metrics = NeuralNetworkManager.test(net, testloader)

#     # JSON schema
#     report = {
#         "validation_loss": loss,
#         "accuracy": metrics["accuracy"],
#         "f1": metrics["f1"],
#         "kappa": metrics["kappa"],
#         "roc_auc": metrics["roc_auc"],
#     }

#     # Save JSON
#     save_metrics(report, out_json_path)
#     print(f"Validation metrics saved to {out_json_path}")

# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python validate.py <in_model_path> <out_json_path>")
#         sys.exit(1)
#     validate(sys.argv[1], sys.argv[2])
##############################################third iteration############################################
############################################################################################################
import os
import sys
import torch
from model import create_model, load_parameters
from data import load_dataset
from fedn.utils.helpers.helpers import save_metrics
from collections import OrderedDict
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, accuracy_score
import numpy as np
import torch.nn.functional as F

# Fetch environment variables
fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
client_id = os.environ.get('CLIENT_ID', 'default_client_id')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetworkManager:
    @staticmethod
    def set_parameters(net, parameters: list):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict if len(v.shape) > 0})
        net.load_state_dict(state_dict, strict=False)
        return net

    @staticmethod
    def test(net, testloader):
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch in testloader:
                images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average='weighted'),
            "kappa": cohen_kappa_score(all_labels, all_preds),
            "roc_auc": roc_auc_score(all_labels, all_probs, multi_class='ovr') if len(set(all_labels)) > 1 else np.nan
        }
        return loss / len(testloader.dataset), metrics

def validate(in_model_path, out_json_path, data_path=None):
    """Validate model.

    :param in_model_path: The path to the input model.
    :type in_model_path: str
    :param out_json_path: The path to save the output JSON to.
    :type out_json_path: str
    :param data_path: The path to the data file.
    :type data_path: str
    """
    # Use environment variable for data path if not provided
    if data_path is None:
        data_path = fedn_data_path
        if data_path is None:
            raise ValueError("Data path must be provided or set in the environment variable 'FEDN_DATA_PATH'.")

    # Load data
    _, testloader = load_dataset(data_path)

    # Load model
    net = load_parameters(in_model_path)
    net.to(DEVICE)

    # Evaluate
    loss, metrics = NeuralNetworkManager.test(net, testloader)

    # JSON schema
    report = {
        "test_loss": loss,               // Using "test_loss" instead of "validation_loss"
        "test_accuracy": metrics["accuracy"],    // Using "test_accuracy" instead of "accuracy"
        "f1": metrics["f1"],
        "kappa": metrics["kappa"],
        "roc_auc": metrics["roc_auc"]
    }

    # Save JSON
    save_metrics(report, out_json_path)
    print(f"Validation metrics saved to {out_json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate.py <in_model_path> <out_json_path>")
        sys.exit(1)
    validate(sys.argv[1], sys.argv[2])
