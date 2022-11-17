



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, MaxPooling1D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping

import os
from os.path import join
import warnings
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, Activation, Concatenate
from tensorflow.keras.layers import Embedding, Conv1D, Conv2D, MaxPooling2D, LSTM, GRU, BatchNormalization, Reshape
from tensorflow.keras.layers import TimeDistributed, Dense, Input, GlobalAveragePooling2D, Bidirectional
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras import models
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import RMSprop, Adadelta, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.decomposition import PCA

from torchsummary import summary

from functools import reduce
from scipy.signal import savgol_filter 
import matplotlib.colors as mcolors


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'





import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR


# ______________________________________________________________________
# ______________________________________________________________________
# ______________________________________________________________________


original_labels = ['study', 'walk', 'sleep', 'idle']
original_features = ['hr', 'gryo_x', 'gyro_y', 'gyro_z']

data_dir = '../HARB4/'
results_dir = "/results"

model_configurations = [{'n_conv_layers': 3, 'n_lstm_layers': 2, 'activation_function': 'relu', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 5, 'lstm_units': 32, 'optimizer': 'Adam', 'learning_rate': 0.0001, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 1, 'n_lstm_layers': 1, 'activation_function': 'sigmoid', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 5, 'lstm_units': 20, 'optimizer': 'Adam', 'learning_rate': 7.000000000000001e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 2, 'n_lstm_layers': 1, 'activation_function': 'relu', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 9, 'lstm_units': 20, 'optimizer': 'Adam', 'learning_rate': 4e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 2, 'n_lstm_layers': 2, 'activation_function': 'relu', 'dropout_rate': 0.1, 'conv_filter': 10, 'conv_kernel_size': 9, 'lstm_units': 32, 'optimizer': 'RMSprop', 'learning_rate': 1e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 2, 'n_lstm_layers': 2, 'activation_function': 'sigmoid', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 9, 'lstm_units': 5, 'optimizer': 'RMSprop', 'learning_rate': 7.000000000000001e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 3, 'n_lstm_layers': 3, 'activation_function': 'tanh', 'dropout_rate': 0.1, 'conv_filter': 5, 'conv_kernel_size': 9, 'lstm_units': 32, 'optimizer': 'Adam', 'learning_rate': 0.0001, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 3, 'n_lstm_layers': 1, 'activation_function': 'relu', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 9, 'lstm_units': 5, 'optimizer': 'RMSprop', 'learning_rate': 1e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 2, 'n_lstm_layers': 3, 'activation_function': 'sigmoid', 'dropout_rate': 0.1, 'conv_filter': 10, 'conv_kernel_size': 18, 'lstm_units': 5, 'optimizer': 'Adam', 'learning_rate': 1e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 1, 'n_lstm_layers': 3, 'activation_function': 'sigmoid', 'dropout_rate': 0.1, 'conv_filter': 5, 'conv_kernel_size': 9, 'lstm_units': 20, 'optimizer': 'SGD', 'learning_rate': 4e-05, 'input_shape': (300, 4), 'n_classes': 4}, {'n_conv_layers': 1, 'n_lstm_layers': 3, 'activation_function': 'sigmoid', 'dropout_rate': 0.1, 'conv_filter': 20, 'conv_kernel_size': 9, 'lstm_units': 5, 'optimizer': 'SGD', 'learning_rate': 4e-05, 'input_shape': (300, 4), 'n_classes': 4}]

parties_classes = [
    [0, 1, 2, 3], 
    [0, 1], 
    [2, 3],
    [0, 2, 3],
    [1, 2, 3],
    [0, 1, 3],
    [0, 1, 2],
    [0, 1, 2, 3],
    [0, 3], 
    [1, 2]
]

title_font = {'fontname':'Arial', 'size':'28', 'color':'black', 'weight':'normal'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'25'}
tick_font = {'fontname':'Arial', 'size':'23'}
legend_font = {'size':'17'}


n_parties = 10
n_samples_per_class = 30

n_alignment =  200
n_iterations = 100 
n_features = 4 
n_classes = 4 
seq_len  = 2000 
offset = 100 
emb_dim = 400 
num_heads = 8









# ______________________________________________________________________
# ______________________________________________________________________
# ______________________________________________________________________





# plot data distribution
def plot_data_distribution(data, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=10, color='blue', edgecolor='black', alpha=0.7)
    ax.set_title(title, **title_font)
    ax.set_xlabel('Number of samples', **axis_font)
    ax.set_ylabel('Number of classes', **axis_font)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.show()