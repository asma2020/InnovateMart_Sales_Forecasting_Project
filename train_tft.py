import os
import warnings
import pickle

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE

warnings.filterwarnings("ignore")

class WrappedTFT(pl.LightningModule):
    def __init__(self, tft: TemporalFusionTransformer):
        super().__init__()
        self.tft = tft
        self._current_fx_name = None
        original_log = getattr(self.tft, "log", None)

        def safe_log(*args, **kwargs):
            if getattr(self.tft, "_current_fx_name", None) is not None and original_log is not None:
                return original_log(*args, **kwargs)
            return None

        self.tft.log = safe_log

    def training_step(self, batch, batch_idx):
        self._current_fx_name = "training_step"
        self.tft._current_fx_name = "training_step"
        self.tft.trainer = self.trainer
        out = self.tft.training_step(batch, batch_idx)
        self._current_fx_name = None
        self.tft._current_fx_name = None
        return out

    def validation_step(self, batch, batch_idx):
        self._current_fx_name = "validation_step"
        self.tft._current_fx_name = "validation_step"
        self.tft.trainer = self.trainer
        out = self.tft.validation_step(batch, batch_idx)
        self._current_fx_name = None
        self.tft._current_fx_name = None
        return out

    def test_step(self, batch, batch_idx):
        self._current_fx_name = "test_step"
        self.tft._current_fx_name = "test_step"
        self.tft.trainer = self.trainer
        out = self.tft.test_step(batch, batch_idx)
        self._current_fx_name = None
        self.tft._current_fx_name = None
        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self._current_fx_name = "predict_step"
        self.tft._current_fx_name = "predict_step"
        self.tft.trainer = self.trainer
        out = self.tft.predict_step(batch, batch_idx, dataloader_idx)
        self._current_fx_name = None
        self.tft._current_fx_name = None
        return out

    def configure_optimizers(self):
        return self.tft.configure_optimizers()

    def on_fit_start(self):
        self.tft.trainer = self.trainer
        self.tft._current_fx_name = None
        return super().on_fit_start()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)

    # ensure categorical dtypes as strings
    df["month"] = df["month"].astype(str)
    df["day_of_week"] = df["day_of_week"].astype(str)
    df["is_weekend"] = df["is_weekend"].astype(str)
    df["store_size"] = df["store_size"].astype(str)
    df["store_id"] = df["store_id"].astype(str)

    # lags & moving averages
    df["sales_lag_7"] = df.groupby("store_id")["daily_sales"].shift(7)
    df["sales_lag_30"] = df.groupby("store_id")["daily_sales"].shift(30)
    df["sales_ma_7"] = df.groupby("store_id")["daily_sales"].transform(
        lambda x: x.rolling(7, min_periods=1).mean()
    )
    df["sales_ma_30"] = df.groupby("store_id")["daily_sales"].transform(
        lambda x: x.rolling(30, min_periods=1).mean()
    )

    df["sales_lag_7"] = df["sales_lag_7"].fillna(df["daily_sales"])
    df["sales_lag_30"] = df["sales_lag_30"].fillna(df["daily_sales"])

    unique_dates = df["date"].sort_values().unique()
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    df["time_idx"] = df["date"].map(date_to_idx)

    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    return df


def make_datasets(df: pd.DataFrame, max_encoder=30, max_prediction=7):
    # choose a train / validation split using time_idx
    train_cut = df["time_idx"].max() - max_prediction - 30
    train = df[df["time_idx"] <= train_cut].copy()
    val = df[df["time_idx"] > train_cut].copy()

    training = TimeSeriesDataSet(
        train,
        time_idx="time_idx",
        target="daily_sales",
        group_ids=["store_id"],
        min_encoder_length=int(max_encoder / 2),
        max_encoder_length=max_encoder,
        min_prediction_length=1,
        max_prediction_length=max_prediction,
        static_categoricals=["store_id", "store_size"],
        static_reals=["city_population"],
        time_varying_known_categoricals=["month", "day_of_week", "is_weekend"],
        time_varying_known_reals=["promotion_active", "days_since_start", "time_idx"],
        time_varying_unknown_reals=[
            "daily_sales",
            "sales_lag_7",
            "sales_lag_30",
            "sales_ma_7",
            "sales_ma_30",
        ],
        target_normalizer=GroupNormalizer(groups=["store_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, val, predict=True, stop_randomization=True
    )
    return training, validation, val  # also return raw validation DataFrame


def train(training, validation, max_epochs=10, gpus=None, skip_sanity_val=True):
    train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # device setup
    if gpus and gpus > 0:
        accelerator = "gpu"
        devices = gpus
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=0.1,
        callbacks=[early_stop],
        logger=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0 if skip_sanity_val else 1,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        log_interval=10,
        reduce_on_plateau_patience=2,
        loss=QuantileLoss(quantiles=[0.5]),
    )

    model = WrappedTFT(tft)
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    return model.tft, trainer


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv("data/simulated_sales.csv")
    df = prepare(df)
    training, validation, val_raw = make_datasets(df, max_encoder=30, max_prediction=7)
    print("Training dataset length:", len(training))
    print("Validation dataset length:", len(validation))

    gpus = 1 if torch.cuda.is_available() else None

    model, trainer = train(
        training, validation, max_epochs=8, gpus=1 if gpus else None, skip_sanity_val=True
    )

    # save model and datasets
    torch.save(model.state_dict(), "models/tft_ckpt.pth")
    with open("models/training_dataset.pkl", "wb") as f:
        pickle.dump(training, f)
    with open("models/validation_dataset.pkl", "wb") as f:
        pickle.dump(validation, f)

    # NEW: save the raw validation DataFrame for Streamlit use
    with open("models/validation_raw.pkl", "wb") as f:
        pickle.dump(val_raw, f)

    print("Saved model, datasets, and raw validation DataFrame to models/")
