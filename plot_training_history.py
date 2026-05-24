import matplotlib.pyplot as plt

def plot_training_history(history, results_dir):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(10,4))
    
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig(f"{results_dir}/training_history.png")
    plt.show()