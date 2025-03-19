import logging
import pickle
from ..architectures.data_sampler import DataSampler
from ..architectures.generator import Generator
import torch
from .utils.sample import sample
import pandas as pd

DATA_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\data\raw\KDDTrain+.csv'
MODEL_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\src\models\gan_generator.pth'
TRANSFORMER_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\src\models\data_transformer.pkl'
LATENT_DIM = 128
GEN_HIDDEN_LAYERS = (256, 256)
DISC_HIDDEN_LAYERS = (256, 256)
DEVICE = 'cpu'

def generate_samples(num_samples: int, batch_size: int = 50):
    """Generates synthetic data using the trained GAN."""
    logging.info("Loading transformer and generator for sample generation...")
    with open(TRANSFORMER_PATH, 'rb') as f:
        transformer = pickle.load(f)
    df = pd.read_csv(DATA_PATH)
    train_data = transformer.transform(df)
    data_sampler = DataSampler(train_data, transformer.output_info_list, True)
    generator = Generator(LATENT_DIM + data_sampler.dim_cond_vec(), GEN_HIDDEN_LAYERS, transformer.output_dimensions)
    generator.load_state_dict(torch.load(MODEL_PATH))
    generator.eval()
    
    logging.info("Generating samples...")
    generated = sample(num_samples, batch_size, LATENT_DIM, DEVICE, data_sampler, generator, transformer)
    return generated

# generated_data = generate_samples(100, 50)