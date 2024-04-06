import numpy as np
import matplotlib.pyplot as plt

train_loss_history_loaded = np.load('train_loss_history.npy')
val_loss_history_loaded = np.load('val_loss_history.npy')

epochs = np.arange(1,13)

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss_history_loaded, 'b', label='Training Loss')
plt.plot(epochs, val_loss_history_loaded, 'r', label='Validation Loss')
plt.title('Training and Validation Loss (Loaded from .npy files)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

train_acc_history_loaded = np.load('train_acc_history.npy')
val_acc_history_loaded = np.load('val_acc_history.npy')

plt.figure(figsize=(10,6))
plt.plot(epochs, train_acc_history_loaded, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_history_loaded, 'r', label='Validation Accuracy')
plt.title('Training and Validation accuracy (Loaded from .npy files)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()




