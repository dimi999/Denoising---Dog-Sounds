import pandas as pd
import tensorflow as tf
import librosa
from sklearn.model_selection import train_test_split
import numpy as np

dataset_df = pd.read_csv('dataset.csv').to_numpy()
sounds_dataset = dataset_df[:, 0]
sr = 0
for i in range(len(sounds_dataset)):
    sounds_dataset[i], sr = librosa.load(f'Noisy-sounds/{sounds_dataset[i]}.wav')

label_dataset = dataset_df[:, 1]
for i in range(len(label_dataset)):
    label_dataset[i], sr = librosa.load(f'Clean-sounds/{label_dataset[i]}.wav')

X_train, X_test, y_train, y_test = train_test_split(sounds_dataset, label_dataset, test_size=0.1, random_state=42)

max_len = 0
for x in X_train:
    if len(x) > max_len:
        max_len = len(x)

for x in X_test:
    if len(x) > max_len:
        max_len = len(x)

for x in y_train:
    if len(x) > max_len:
        max_len = len(x)

for x in y_test:
    if len(x) > max_len:
        max_len = len(x)


def pad_vector(vec):
    for i in range(len(vec)):
        vec[i] = np.pad(vec[i], (0, max_len - len(vec[i])))
        # vec[i] = np.asarray(vec[i]).astype(np.float32)
    return vec


X_train = pad_vector(X_train)
X_test = pad_vector(X_test)
y_train = pad_vector(y_train)
y_test = pad_vector(y_test)

X_train_good = np.ones((len(X_train), max_len))
X_test_good = np.ones((len(X_test), max_len))
y_train_good = np.ones((len(y_train), max_len))
y_test_good = np.ones((len(y_test), max_len))

for i in range(len(X_train_good)):
    X_train_good[i] = X_train[i]
for i in range(len(X_test_good)):
    X_test_good[i] = X_test[i]
for i in range(len(y_train_good)):
    y_train_good[i] = y_train[i]
for i in range(len(y_test_good)):
    y_test_good[i] = y_test[i]

input_dim = max_len  # Dimensionality of input data (e.g., length of audio signal)
latent_dim = 128  # Dimensionality of the latent space

# Define the architecture of the autoencoder
class DenoisingAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x, *kwargs):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Define the hyperparameters
epochs = 5
batch_size = 32

# Define the model
autoencoder = DenoisingAutoencoder(latent_dim)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error loss for audio reconstruction

# Train the model
autoencoder.fit(X_train_good, y_train_good, epochs=epochs, batch_size=batch_size, validation_data=(X_test_good, y_test_good))

test, sr = librosa.load(f'Noisy-sounds/Sound12.wav')
