import os
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader, random_split
from ..preprocessing.preprocess import Preprocessor
from .train_autoencoder import train_cae
from ..architectures.auto_encoder import ContractiveAutoEncoder
from ..architectures.SGAE_GC import SCAE_GC
from .train_SGAE_GC import train_scae_gc_model
from .utils.datasets import CustomDataset, TensorDataset
import json
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = "C:/Users/Vishruth V Srivatsa/OneDrive/Desktop/IDS/data/raw/KDDTrain+.csv"
OUTPUT_PATH = "C:/Users/Vishruth V Srivatsa/OneDrive/Desktop/IDS/src/IDS/output"
MODEL_SAVE_PATH = "C:/Users/Vishruth V Srivatsa/OneDrive/Desktop/IDS/src/models"
MAPPING_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "label_mapping.json")
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "preprocessor.pkl")
BATCH_SIZE = 32
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = 'cpu'

RIGHT_SKEWED = ['0', '491', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.18', '2', '2.1', '0.00', '0.00.1', '0.00.2']
LEFT_SKEWED = ['20', '150', '1.00']
TYPES = ['normal', 'neptune', 'warezclient', 'portsweep', 'smurf', 
         'satan', 'ipsweep', 'nmap', 'imap', 'back', 'multihop', 'warezmaster']

def save_model_weights(models, model_names, save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        for model, name in zip(models, model_names):
            path = os.path.join(save_dir, f"{name}.pth")
            torch.save(model.state_dict(), path)
            logging.info(f"Model {name} saved successfully at {path}")
    except Exception as e:
        logging.error(f"Error in saving models: {e}")
        raise

def save_preprocessor(preprocessor, save_path):
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info(f"Preprocessor saved successfully at {save_path}")
    except Exception as e:
        logging.error(f"Error in saving preprocessor: {e}")
        raise

def save_mapping(mapping, save_path):
    try:
        with open(save_path, 'w') as f:
            json.dump(mapping, f, indent=4)
        logging.info(f"Label mapping saved successfully at {save_path}")
    except Exception as e:
        logging.error(f"Error in saving label mapping: {e}")
        raise

try:
    preprocessor = Preprocessor(OUTPUT_PATH)
    df = pd.read_csv(DATA_PATH)
    preprocessor.load_data(df, 'normal')
    preprocessor.process(LEFT_SKEWED, RIGHT_SKEWED)
    final_features, labels = preprocessor.df, preprocessor.labels
    logging.info(f"Data loaded and processed successfully. Shape: {final_features.shape}")
except Exception as e:
    logging.error(f"Error in preprocessing: {e}")
    raise

def map_types_to_numbers(series, types):
    type_to_number = {type_name: i for i, type_name in enumerate(types)}
    return series.map(type_to_number).values, type_to_number

mapped_result, mapping = map_types_to_numbers(pd.Series(labels), TYPES)

dataset = CustomDataset(final_features, mapped_result)
total_size = len(dataset)
train_size = int(TRAIN_RATIO * total_size)
test_size = total_size - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
logging.info("DataLoader created successfully.")

def create_dataset(cae, loader, device):
    cae.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            h, _ = cae(features)
            features_list.append(h.cpu())
            labels_list.append(labels.cpu())
    return DataLoader(TensorDataset(torch.cat(features_list), torch.cat(labels_list)), batch_size=BATCH_SIZE, shuffle=True)

# Initialize and train autoencoders
CAE_LAYERS = [(37, 80), (80, 40), (40, 20)]
autoencoders = [ContractiveAutoEncoder(in_dim, out_dim) for in_dim, out_dim in CAE_LAYERS]

try:
    data_loader = train_loader
    trained_autoencoders = []
    for cae in autoencoders:
        trained_cae = train_cae(cae, data_loader, EPOCHS, LEARNING_RATE, DEVICE)
        trained_autoencoders.append(trained_cae)
        data_loader = create_dataset(trained_cae, data_loader, DEVICE)
    logging.info("All autoencoders trained successfully.")
except Exception as e:
    logging.error(f"Error in training autoencoders: {e}")
    raise

try:
    scae_gc = SCAE_GC(37, *trained_autoencoders, 20, 20)
    trained_model = train_scae_gc_model(scae_gc, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
    logging.info("SCAE-GC model trained successfully.")
except Exception as e:
    logging.error(f"Error in training SCAE-GC model: {e}")
    raise

model_list = trained_autoencoders + [trained_model]
model_names = ["CAE1", "CAE2", "CAE3", "SCAE_GC"]
save_model_weights(model_list, model_names, MODEL_SAVE_PATH)
save_preprocessor(preprocessor, PREPROCESSOR_SAVE_PATH)
save_mapping(mapping, MAPPING_SAVE_PATH)
