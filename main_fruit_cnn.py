from src.preprocess import create_generators
from src.train_model import build_cnn_model
from src.visualize import plot_training_history
import os

# Paths
data_dir = 'data'
results_dir = 'results'
models_dir = 'models'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# Parameters
img_size = (128,128)
batch_size = 32

# Generators
train_gen, val_gen = create_generators(data_dir, img_size=img_size, batch_size=batch_size)

# Build model
model = build_cnn_model(input_shape=(128,128,3), num_classes=train_gen.num_classes)

# Train model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10
)

# Save model
model.save(f"{models_dir}/fruit_cnn_model.h5")

# Plot results
plot_training_h