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
# Check if files exist
# -----------------------------
def check_files():
    files_needed = [DATA_PATH, TRAIN_DS, VAL_DS, VAL_RAW, MODEL_STATE]
    missing_files = [f for f in files_needed if not os.path.exists(f)]
    return missing_files

missing = check_files()
if missing:
    st.error("⚠️ فایل‌های زیر وجود ندارند:")
    for file in missing:
        st.write(f"❌ {file}")
    st.info("لطفاً ابتدا train_tft.py را اجرا کنید و فایل‌های لازم را در پوشه‌های مربوطه قرار دهید.")
    st.stop()

# -----------------------------
# Helpers: load data & artifacts
# -----------------------------
@st.cache_data
def load_data(path=DATA_PATH):
    try:
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
        df["store_id"] = df["store_id"].astype(str)
        return df
    except Exception as e:
        st.error(f"خطا در بارگذاری داده‌ها: {e}")
        return None

@st.cache_resource
def load_artifacts():
    try:
        with open(TRAIN_DS, "rb") as f:
            training = pickle.load(f)
        with open(VAL_DS, "rb") as f:
            validation = pickle.load(f)
        with open(VAL_RAW, "rb") as f:
            validation_raw = pickle.load(f)

        # Handle validation_raw format
        if not isinstance(validation_raw, pd.DataFrame):
            if isinstance(validation_raw, (list, tuple)) and len(validation_raw) > 0:
                for el in validation_raw:
                    if isinstance(el, pd.DataFrame):
                        validation_raw = el
                        break

        # Load model
        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=1e-3,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=None,
        )
        state = torch.load(MODEL_STATE, map_location=torch.device("cpu"))
        tft.load_state_dict(state)
        return training, validation, validation_raw, tft
    except Exception as e:
        st.error(f"خطا در بارگذاری مدل: {e}")
        return None

# -----------------------------
# Load everything
# -----------------------------
df = load_data()
if df is None:
    st.stop()

artifacts = load_artifacts()
if artifacts is None:
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

if not isinstance(validation_raw, pd.DataFrame):
    st.error("validation_raw موجود نیست یا فرمت آن پشتیبانی نمی‌شود.")
    st.stop()

val_store = validation_raw[validation_raw["store_id"].astype(str) == str(sel)].copy()

if val_store.empty:
    st.info("برای این فروشگاه دادهٔ اعتبارسنجی وجود ندارد.")
else:
    try:
        # Build dataset & dataloader for this store
        pred_ds = TimeSeriesDataSet.from_dataset(training, val_store, predict=True, stop_randomization=True)
        dl = pred_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        # Get predictions
        with st.spinner("در حال پیش‌بینی..."):
            raw_output = tft.predict(dl, mode="raw", return_x=True)

        # Extract predictions tensor
        try:
            preds_tensor = raw_output.output.prediction
        except:
            preds_tensor = raw_output.output

        if preds_tensor.ndim == 3 and preds_tensor.shape[-1] == 1:
            preds_tensor = preds_tensor.squeeze(-1)

        predictions = preds_tensor.detach().cpu().numpy() if hasattr(preds_tensor, "detach") else np.array(preds_tensor)

        x = raw_output.x
        unique_dates = np.array(df["date"].sort_values().unique())
        max_pred_len = pred_ds.max_prediction_length
        encoder_length = pred_ds.max_encoder_length

        rows = []
        decoder_time_idx = x.get("decoder_time_idx", None)
        encoder_time_idx = x.get("encoder_time_idx", None)

        for i in range(len(predictions)):
            if decoder_time_idx is not None:
                d0 = decoder_time_idx[i, 0].item() if hasattr(decoder_time_idx, "shape") else int(decoder_time_idx[i][0])
                start_time_idx = d0 - encoder_length
            elif encoder_time_idx is not None:
                enc_last = encoder_time_idx[i, -1].item() if hasattr(encoder_time_idx, "shape") else int(encoder_time_idx[i][-1])
                start_time_idx = enc_last - encoder_length + 1
            else:
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

        st.write("نمونهٔ خروجی پیش‌بینی):")
        st.dataframe(pred_df.head(50))

        # Prepare overlay chart
        actuals = store_df[["date", "daily_sales"]].rename(columns={"daily_sales": "sales"}).copy()
        actuals["type"] = "actual"
        actuals["horizon"] = 0

        preds_clean = pred_df.dropna(subset=["date"]).copy()
        preds_clean = preds_clean.rename(columns={"predicted_sales": "sales"})
        preds_clean["type"] = "predicted"

        actuals["date"] = pd.to_datetime(actuals["date"])
        preds_clean["date"] = pd.to_datetime(preds_clean["date"])

        combined = pd.concat([
            actuals[["date", "sales", "type", "horizon"]],
            preds_clean[["date", "sales", "type", "horizon"]]
        ], ignore_index=True)

        combined = combined.sort_values("date").reset_index(drop=True)

        overlay_chart = alt.Chart(combined).mark_line(opacity=0.8).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("sales:Q", title="Sales"),
            color=alt.Color("type:N", title="Legend",
                            scale=alt.Scale(domain=["actual", "predicted"], range=["#1f77b4", "#ff7f0e"])),
            detail="horizon:N",
            tooltip=["date:T", "sales:Q", "type:N", "horizon:N"]
        ).properties(
            width=900,
            height=420,
            title=f" store_id={sel} فروش تاریخی و پیش‌بینی‌شده برای"
        ).interactive()

        st.altair_chart(overlay_chart, use_container_width=True)

    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
        st.write("جزئیات خطا:", str(e))

# -----------------------------
# -----------------------------
# Variable Importance (Safe Version)
# -----------------------------
st.markdown("---")
st.subheader("اهمیت متغیرها")

try:
    # Simple version - List of variables
    st.write("### نسخه ساده: لیست متغیرهای ورودی")
    
    # Continuous variables
    continuous_vars = training.reals if hasattr(training, 'reals') else []
    st.write("**متغیرهای پیوسته:**", continuous_vars)
    
    # Categorical variables
    categorical_vars = training.categoricals if hasattr(training, 'categoricals') else []
    st.write("**متغیرهای طبقه‌ای:**", categorical_vars if categorical_vars else "هیچ متغیر طبقه‌ای تعریف نشده")
    
    # Time-varying known reals
    time_varying_known = training.time_varying_known_reals if hasattr(training, 'time_varying_known_reals') else []
    st.write("**متغیرهای پیوستهٔ متغیر با زمان:**", time_varying_known)
    
    # Time-varying unknown reals
    time_varying_unknown = training.time_varying_unknown_reals if hasattr(training, 'time_varying_unknown_reals') else []
    st.write("**متغیرهای ناشناختهٔ متغیر با زمان:**", time_varying_unknown)
    
    st.info("💡 نسخه ساده فقط لیست متغیرها را نمایش می‌دهد.")
    
    # Advanced version - Feature importance not available
    st.write("### نسخه پیشرفته: اهمیت متغیرها")
    st.warning("⚠️ در حال حاضر امکان محاسبه اهمیت متغیرها وجود ندارد!")
    
except Exception as e:
    st.error(f"خطا در نمایش اهمیت متغیرها: {str(e)}")

st.success("✅ اپلیکیشن با موفقیت بارگذاری شد!")



