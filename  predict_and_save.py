# predict_and_save.py
import pickle
import torch
import pandas as pd
import numpy as np
import os
from pytorch_forecasting import TemporalFusionTransformer

# -------------------
# Paths
# -------------------
MODEL_STATE = "models/tft_ckpt.pth"
TRAIN_DS = "models/training_dataset.pkl"
VAL_DS = "models/validation_dataset.pkl"
DATA = "data/simulated_sales.csv"

# -------------------
# Load datasets
# -------------------
with open(TRAIN_DS, "rb") as f:
    training = pickle.load(f)
with open(VAL_DS, "rb") as f:
    validation = pickle.load(f)

# -------------------
# Recreate model architecture from dataset
# -------------------
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=1e-3,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    loss=None,  # not needed for inference
)

# -------------------
# Load model weights
# -------------------
state = torch.load(MODEL_STATE, map_location=torch.device("cpu"))
tft.load_state_dict(state)

# -------------------
# Predict (raw mode to inspect + return_x for alignment)
# -------------------
raw_output = tft.predict(validation, mode="raw", return_x=True)

# Extract the tensor from the Lightning Output object
predictions_tensor = raw_output.output.prediction  # shape: (n_windows, max_pred_len, n_quantiles)

# If only one quantile, remove that dimension
if predictions_tensor.shape[-1] == 1:
    predictions_tensor = predictions_tensor.squeeze(-1)  # -> (n_windows, max_pred_len)

# Convert to numpy
predictions = predictions_tensor.detach().cpu().numpy()

# -------------------
# Save raw numpy predictions for Streamlit
# -------------------
os.makedirs("predictions", exist_ok=True)
np.save("predictions/val_preds.npy", predictions)
print(f"Saved predictions to predictions/val_preds.npy with shape {predictions.shape}")

# -------------------
# Optionally, save actuals and decoder start dates for alignment in Streamlit
# -------------------
x = raw_output.x
decoder_target = x["decoder_target"].detach().cpu().numpy()  # actuals for prediction horizon
index = x["decoder_time_idx"].detach().cpu().numpy()         # time index for alignment

np.save("predictions/val_actuals.npy", decoder_target)
np.save("predictions/val_timeidx.npy", index)
print(f"Saved actuals to predictions/val_actuals.npy with shape {decoder_target.shape}")
