import logging
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from ..preprocessing.data_transformer import DataTransformer
from ..architectures.data_sampler import DataSampler
from ..architectures.generator import Generator
from ..architectures.discriminator import Discriminator
from .utils.activate import apply_activate
from .utils.cond_loss import cond_loss
from .utils.sample import sample
from torch import optim

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATA_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\data\raw\KDDTrain+.csv'
DISCRETE_COLUMNS = ['tcp', 'ftp_data', 'SF', 'normal']
BATCH_SIZE = 500
EPOCHS = 1
LATENT_DIM = 128
GEN_HIDDEN_LAYERS = (256, 256)
DISC_HIDDEN_LAYERS = (256, 256)
PAC = 10
LR = 2e-4
BETAS = (0.5, 0.9)
WEIGHT_DECAY = 1e-6
GRADIENT_PENALTY = 10
DEVICE = 'cpu'

try:
    logging.info("Loading dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logging.info("Dataset loaded successfully with shape: %s", df.shape)
except Exception as e:
    logging.error("Failed to load dataset: %s", e)
    raise

try:
    logging.info("Initializing DataTransformer and fitting dataset")
    transformer = DataTransformer()
    transformer.fit(df, DISCRETE_COLUMNS)
    logging.info("Finished fitting data. Transforming...")
    train_data = transformer.transform(df)
    data_sampler = DataSampler(train_data, transformer.output_info_list, True)
    data_dim = transformer.output_dimensions
    logging.info("Data transformation completed. Data dimensions: %d", data_dim)
except Exception as e:
    logging.error("Error in data transformation: %s", e)
    raise

generator = Generator(LATENT_DIM + data_sampler.dim_cond_vec(), GEN_HIDDEN_LAYERS, data_dim)
discriminator = Discriminator(data_dim + data_sampler.dim_cond_vec(), DISC_HIDDEN_LAYERS, pac=PAC)

optimizerG = optim.Adam(generator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
optimizerD = optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)

mean = torch.zeros(BATCH_SIZE, LATENT_DIM, device=DEVICE)
std = mean + 1
loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Discriminator Loss'])
steps_per_epoch = max(len(train_data) // BATCH_SIZE, 1)

logging.info("Starting training with %d epochs and batch size %d", EPOCHS, BATCH_SIZE)
epoch_iterator = tqdm(range(EPOCHS), desc="Gen. (0.00) | Discrim. (0.00)")
for epoch in epoch_iterator:
    try:
        logging.info("Epoch %d/%d started", epoch + 1, EPOCHS)
        for step in range(steps_per_epoch):
            for _ in range(2):
                fakez = torch.normal(mean=mean, std=std)
                condvec = data_sampler.sample_condvec(BATCH_SIZE)
                if condvec is None:
                    real = data_sampler.sample_data(train_data, BATCH_SIZE, None, None)
                    c1, c2 = None, None
                else:
                    c1, m1, col, opt = map(torch.from_numpy, condvec)
                    fakez = torch.cat([fakez, c1], dim=1)
                    perm = np.random.permutation(BATCH_SIZE)
                    real = data_sampler.sample_data(train_data, BATCH_SIZE, col[perm], opt[perm])
                    c2 = c1[perm]
                fake = generator(fakez)
                fakeact = apply_activate(fake, transformer)
                real = torch.tensor(real, dtype=torch.float32)
                real_cat = torch.cat([real, c2], dim=1) if c1 is not None else real
                fake_cat = torch.cat([fakeact, c1], dim=1) if c1 is not None else fakeact
                y_real = discriminator(real_cat)
                y_fake = discriminator(fake_cat)
                pen = discriminator.calc_gradient_penalty(real_cat, fake_cat, DEVICE, GRADIENT_PENALTY)
                loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                optimizerD.zero_grad(set_to_none=True)
                (pen + loss_d).backward()
                optimizerD.step()
            
            fakez = torch.normal(mean=mean, std=std)
            condvec = data_sampler.sample_condvec(BATCH_SIZE)
            if condvec is None:
                c1, m1 = None, None
            else:
                c1, m1, _, _ = map(lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE), condvec)
                fakez = torch.cat([fakez, c1], dim=1)
            fake = generator(fakez)
            fakeact = apply_activate(fake, transformer)
            y_fake = discriminator(torch.cat([fakeact, c1], dim=1) if c1 is not None else fakeact)
            cross_entropy = cond_loss(fake, c1, m1, transformer) if condvec is not None else 0
            loss_g = -torch.mean(y_fake) + cross_entropy
            optimizerG.zero_grad(set_to_none=True)
            loss_g.backward()
            optimizerG.step()
        
        generator_loss = loss_g.item()
        discriminator_loss = loss_d.item()
        loss_values = pd.concat([loss_values, pd.DataFrame({'Epoch': [epoch], 'Generator Loss': [generator_loss], 'Discriminator Loss': [discriminator_loss]})]).reset_index(drop=True)
        epoch_iterator.set_description(f"Gen. ({generator_loss:.2f}) | Discrim. ({discriminator_loss:.2f})")
        logging.info("Epoch %d completed. Generator Loss: %.4f, Discriminator Loss: %.4f", epoch + 1, generator_loss, discriminator_loss)
    except Exception as e:
        logging.error("Error during training at epoch %d: %s", epoch, e)
        raise

try:
    logging.info("Generating synthetic samples...")
    generated = sample(200, 50, LATENT_DIM, DEVICE, data_sampler, generator, transformer)
    logging.info("Sample generation successful. First 5 rows:\n%s", generated.head())
except Exception as e:
    logging.error("Error during sample generation: %s", e)
    raise
