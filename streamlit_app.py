# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import altair as alt
import torch
import os

# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(layout="wide", page_title="InnovateMart Forecast")
st.title("InnovateMart — پیش‌بینی فروش روزانه")

# -----------------------------
# File Paths
# -----------------------------
DATA_PATH = "data/simulated_sales.csv"
TRAIN_DS = "models/training_dataset.pkl"
VAL_DS = "models/validation_dataset.pkl"
VAL_RAW = "models/validation_raw.pkl"
MODEL_STATE = "models/tft_ckpt.pth"

# -----------------------------
# Helpers: load data & artifacts
# -----------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    # ensure consistent dtypes used by TimeSeriesDataSet
    df["store_id"] = df["store_id"].astype(str)
    return df

@st.cache_resource
def load_artifacts():
    files_needed = [TRAIN_DS, VAL_DS, VAL_RAW, MODEL_STATE]
    if not all(os.path.exists(p) for p in files_needed):
        return None

    with open(TRAIN_DS, "rb") as f:
        training = pickle.load(f)
    with open(VAL_DS, "rb") as f:
        validation = pickle.load(f)
    with open(VAL_RAW, "rb") as f:
        validation_raw = pickle.load(f)

    # If validation_raw saved as (x, y) or list/tuple with DataFrame inside, try to extract DF
    if not isinstance(validation_raw, pd.DataFrame):
        if isinstance(validation_raw, (list, tuple)) and len(validation_raw) > 0:
            # pick the first element that is a DataFrame
            for el in validation_raw:
                if isinstance(el, pd.DataFrame):
                    validation_raw = el
                    break

    # instantiate model (structure must match the trained model)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=None,  # not used for inference
    )
    state = torch.load(MODEL_STATE, map_location=torch.device("cpu"))
    tft.load_state_dict(state)
    return training, validation, validation_raw, tft

# -----------------------------
# Load everything
# -----------------------------
df = load_data()
artifacts = load_artifacts()
if artifacts is None:
    st.error("مدل یا داده‌های آموزش وجود ندارد. ابتدا train_tft.py را اجرا کنید و فایل‌های لازم را در پوشه models/ قرار دهید.")
    st.stop()

training, validation, validation_raw, tft = artifacts

# -----------------------------
# UI: store selection
# -----------------------------
store_ids = sorted(df["store_id"].unique().astype(str).tolist())
sel = st.selectbox("انتخاب store_id:", store_ids)

# -----------------------------
# Historical Sales chart
# -----------------------------
store_df = df[df["store_id"].astype(str) == str(sel)].copy()
st.subheader("فروش تاریخی")
if store_df.empty:
    st.info("برای این فروشگاه دادهٔ تاریخی وجود ندارد.")
else:
    chart_hist = alt.Chart(store_df).mark_line().encode(
        x=alt.X("date:T", title="تاریخ"),
        y=alt.Y("daily_sales:Q", title="فروش")
    ).properties(height=300)
    st.altair_chart(chart_hist, use_container_width=True)

# -----------------------------
# Forecast on validation set for selected store
# -----------------------------
st.subheader("پیش‌بینی روی مجموعهٔ اعتبارسنجی")

# Ensure validation_raw is a DataFrame with store_id column
if not isinstance(validation_raw, pd.DataFrame):
    st.error("validation_raw موجود نیست یا فرمت آن پشتیبانی نمی‌شود.")
    st.stop()

# filter validation rows for the selected store
val_store = validation_raw[validation_raw["store_id"].astype(str) == str(sel)].copy()

if val_store.empty:
    st.info("برای این فروشگاه دادهٔ اعتبارسنجی وجود ندارد.")
else:
    # build dataset & dataloader for this store
    pred_ds = TimeSeriesDataSet.from_dataset(training, val_store, predict=True, stop_randomization=True)
    dl = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

    # get raw output which contains tensor and x
    raw_output = tft.predict(dl, mode="raw", return_x=True)

    # extract predictions tensor: (n_windows, max_pred_len, n_quantiles)
    # raw_output.output may be a small wrapper: access .prediction if present
    try:
        preds_tensor = raw_output.output.prediction
    except Exception:
        # fallback if directly a tensor
        preds_tensor = raw_output.output

    # If a single quantile, squeeze last dim
    if preds_tensor.ndim == 3 and preds_tensor.shape[-1] == 1:
        preds_tensor = preds_tensor.squeeze(-1)  # -> (n_windows, max_pred_len)

    # Ensure numpy
    predictions = preds_tensor.detach().cpu().numpy() if hasattr(preds_tensor, "detach") else np.array(preds_tensor)

    # input dict for alignment
    x = raw_output.x

    # map time_idx to real dates using original df.unique dates
    unique_dates = np.array(df["date"].sort_values().unique())

    max_pred_len = pred_ds.max_prediction_length
    encoder_length = pred_ds.max_encoder_length

    rows = []
    # decoder_time_idx shape: (n_windows, max_pred_len)
    decoder_time_idx = x.get("decoder_time_idx", None)
    encoder_time_idx = x.get("encoder_time_idx", None)

    # We'll compute start_time_idx robustly:
    # If decoder_time_idx available, use its first element; otherwise fallback to encoder_time_idx last + 1
    for i in range(len(predictions)):
        if decoder_time_idx is not None:
            # decoder_time_idx may be a tensor
            d0 = decoder_time_idx[i, 0].item() if hasattr(decoder_time_idx, "shape") else int(decoder_time_idx[i][0])
            # start_time_idx such that encoder_length steps come before decoder
            start_time_idx = d0 - encoder_length
        elif encoder_time_idx is not None:
            # use last encoder time idx as base for decoder start
            enc_last = encoder_time_idx[i, -1].item() if hasattr(encoder_time_idx, "shape") else int(encoder_time_idx[i][-1])
            start_time_idx = enc_last - encoder_length + 1
        else:
            # fallback: use time_idx from dataset mapping (less robust)
            start_time_idx = 0

        pred_vals = predictions[i]
        for h in range(max_pred_len):
            abs_time_idx = start_time_idx + encoder_length + h
            date = unique_dates[abs_time_idx] if 0 <= abs_time_idx < len(unique_dates) else None
            rows.append({
                "window": i,
                "horizon": h + 1,
                "predicted_sales": float(pred_vals[h]),
                "date": pd.to_datetime(date) if date is not None else pd.NaT
            })

    pred_df = pd.DataFrame(rows)

    st.write("نمونهٔ خروجی پیش‌بینی (هر ردیف = پنجره، افق پیش‌بینی):")
    st.dataframe(pred_df.head(50))

    # === Prepare overlay with all horizons plotted separately ===
    # actuals for the selected store
    actuals = store_df[["date", "daily_sales"]].rename(columns={"daily_sales": "sales"}).copy()
    actuals["type"] = "actual"
    actuals["horizon"] = 0  # actuals horizon 0

    preds_clean = pred_df.dropna(subset=["date"]).copy()
    preds_clean = preds_clean.rename(columns={"predicted_sales": "sales"})
    preds_clean["type"] = "predicted"

    # ensure date dtypes
    actuals["date"] = pd.to_datetime(actuals["date"])
    preds_clean["date"] = pd.to_datetime(preds_clean["date"])

    # include horizon column in preds_clean (already present)
    combined = pd.concat([
        actuals[["date", "sales", "type", "horizon"]],
        preds_clean[["date", "sales", "type", "horizon"]]
    ], ignore_index=True)

    # Sort by date for nice plotting
    combined = combined.sort_values("date").reset_index(drop=True)

    # Plot: actual line (blue) and predicted lines for each horizon (orange scale)
    overlay_chart = alt.Chart(combined).mark_line(opacity=0.8).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("sales:Q", title="Sales"),
        color=alt.Color("type:N", title="Legend",
                        scale=alt.Scale(domain=["actual", "predicted"], range=["#1f77b4", "#ff7f0e"])),
        detail="horizon:N",  # ensures a separate line per horizon
        tooltip=["date:T", "sales:Q", "type:N", "horizon:N"]
    ).properties(
        width=900,
        height=420,
        title=f"فروش تاریخی و پیش‌بینی‌شده برای store_id={sel} (تمام افق‌ها)"
    ).interactive()

    st.altair_chart(overlay_chart, use_container_width=True)

# -----------------------------
# Variable Importance (best-effort)
# -----------------------------
st.markdown("---")
st.subheader("اهمیت متغیرها (Permutation importance approximation)")
try:
    # pred_ds may not exist if no val_store; guard that
    dl_vi = pred_ds.to_dataloader(train=False, batch_size=64) if "pred_ds" in locals() else None
    if dl_vi is not None:
        vi = tft.calculate_variable_importance(dl_vi)
        vi_df = vi.reset_index().rename(columns={"index": "feature", 0: "importance"}).sort_values("importance", ascending=False)
        st.bar_chart(vi_df.set_index("feature")["importance"])
    else:
        st.info("دیتاست اعتبارسنجی برای محاسبهٔ اهمیت متغیرها موجود نیست.")
except Exception as e:
    st.write("محاسبهٔ اهمیت متغیرها ناموفق بود:", e)

st.write("پایان.")
