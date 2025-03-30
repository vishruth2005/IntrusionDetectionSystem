import os

DATA_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\backend\data\raw\KDDTrain+.csv'
MODEL_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\backend\src\models\gan_generator.pth'
TRANSFORMER_PATH = r'C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\backend\src\models\data_transformer.pkl'
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

OUTPUT_PATH = r"C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\backend\src\IDS\output"
MODEL_SAVE_PATH = r"C:\Users\Vishruth V Srivatsa\OneDrive\Desktop\IDS\backend\src\models"
MAPPING_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "label_mapping.json")
PREPROCESSOR_SAVE_PATH = os.path.join(MODEL_SAVE_PATH, "preprocessor.pkl")
IDS_BATCH_SIZE = 32
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
EPOCHS = 1
LEARNING_RATE = 0.001
DEVICE = 'cpu'

RIGHT_SKEWED = ['0', '491', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.18', '2', '2.1', '0.00', '0.00.1', '0.00.2']
LEFT_SKEWED = ['20', '150', '1.00']
TYPES = ['normal', 'neptune', 'warezclient', 'portsweep', 'smurf', 
         'satan', 'ipsweep', 'nmap', 'imap', 'back', 'multihop', 'warezmaster']