# import os
# import pickle
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, Dataset
# fedn_data_path = os.environ.get('FEDN_DATA_PATH', '/var/data')
# client_id= os.environ.get('client_id')
# class CustomImageDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img = self.images[idx]
#         label = self.labels[idx].argmax().item()
#         img = Image.fromarray((img * 255).astype(np.uint8))
#         if self.transform:
#             img = self.transform(img)
#         return img, label

# def load_pyp_file(filepath):
#     with open(filepath, 'rb') as f:
#         data = pickle.load(f)
#     return data

# def load_dataset(fedn_data_path, client_id, batch_size=32):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
    
#     #client_path = os.path.join(fedn_data_path, f'client{client_id}')
#     #client_path = os.path.join(data_path, f'client{client_id}')
#     try:
#         train_x = load_pyp_file(os.path.join(fedn_data_path, 'trainx.pyp'))
#         train_y = load_pyp_file(os.path.join(fedn_data_path, 'trainy.pyp'))
#         test_x = load_pyp_file(os.path.join(fedn_data_path, 'testx.pyp'))
#         test_y = load_pyp_file(os.path.join(fedn_data_path, 'testy.pyp'))

#         train_dataset = CustomImageDataset(np.array(train_x), np.array(train_y), transform=transform)
#         test_dataset = CustomImageDataset(np.array(test_x), np.array(test_y), transform=transform)

#         trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         return trainloader, testloader
#     except FileNotFoundError:
#         print(f"Data files not found for client {client_id}. Check the data directory structure.")
#         return None, None

# if __name__ == "__main__":
#     # Example of loading data for client 0 (just for testing purposes)
#     trainloader, testloader = load_dataset(fedn_data_path, client_id)
#     print(f"Trainloader: {len(trainloader.dataset)} samples")
#     print(f"Testloader: {len(testloader.dataset)} samples")
import os
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms

FEDN_DATA_PATH = os.getenv('FEDN_DATA_PATH', './data')
client_id= os.environ.get('client_id')
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx].argmax().item()
        img = Image.fromarray((img * 255).astype(np.uint8))  # Scale to [0, 255] and convert to uint8
        if self.transform:
            img = self.transform(img)
        return {"img": img, "label": label}

def load_pyp_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    #client_path = os.path.join(FEDN_DATA_PATH, f'client{client_id}')
    train_x = load_pyp_file(os.path.join(FEDN_DATA_PATH, 'trainx.pyp'))
    train_y = load_pyp_file(os.path.join(FEDN_DATA_PATH, 'trainy.pyp'))
    test_x = load_pyp_file(os.path.join(FEDN_DATA_PATH, 'testx.pyp'))
    test_y = load_pyp_file(os.path.join(FEDN_DATA_PATH, 'testy.pyp'))

    train_dataset = CustomImageDataset(np.array(train_x), np.array(train_y), transform=transform)
    test_dataset = CustomImageDataset(np.array(test_x), np.array(test_y), transform=transform)

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return trainloader, testloader
