# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
import altair as alt
import torch
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
# Feature Importance Functions
# -----------------------------
def calculate_permutation_importance(model, dataloader, n_repeats=5):
    """Calculate permutation importance for model features"""
    
    # Get baseline predictions and performance
    baseline_preds = []
    true_values = []
    
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            # FIX: Access the prediction tensor from the model's Output object
            pred_output = model(x)
            pred_tensor = pred_output.prediction
            baseline_preds.append(pred_tensor.cpu().numpy())
            true_values.append(y[0].cpu().numpy())  # target is first element
    
    baseline_preds = np.concatenate(baseline_preds, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    
    # Calculate baseline MAE
    baseline_mae = mean_absolute_error(true_values.flatten(), baseline_preds.flatten())
    
    # Get feature names
    feature_names = []
    if hasattr(dataloader.dataset, 'reals'):
        feature_names.extend(dataloader.dataset.reals)
    if hasattr(dataloader.dataset, 'categoricals'):
        feature_names.extend(dataloader.dataset.categoricals)
    
    importance_scores = {}
    
    st.write(f"📊 محاسبه اهمیت متغیرها (MAE پایه: {baseline_mae:.4f})")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate importance for each feature
    # Permuting only a subset of features for demonstration
    features_to_check = [f for f in feature_names if f not in ['date']] 
    
    for feat_idx, feature_name in enumerate(features_to_check):
        status_text.text(f"در حال محاسبه اهمیت برای: {feature_name}")
        
        importance_values = []
        
        for repeat in range(n_repeats):
            permuted_preds = []
            
            model.eval()
            with torch.no_grad():
                for x, y in dataloader:
                    # Create a deep copy of the input dictionary to avoid side effects
                    x_permuted = {key: value.clone() for key, value in x.items()}
                    
                    # Permute the feature if it exists in continuous variables
                    if feature_name in dataloader.dataset.reals:
                        feat_pos = dataloader.dataset.reals.index(feature_name)
                        if 'encoder_cont' in x_permuted:
                            perm_indices = torch.randperm(x_permuted['encoder_cont'].shape[0])
                            x_permuted['encoder_cont'][:, :, feat_pos] = x_permuted['encoder_cont'][perm_indices, :, feat_pos]
                        if 'decoder_cont' in x_permuted:
                            perm_indices = torch.randperm(x_permuted['decoder_cont'].shape[0])
                            x_permuted['decoder_cont'][:, :, feat_pos] = x_permuted['decoder_cont'][perm_indices, :, feat_pos]
                    
                    # Permute for categorical variables (requires a different approach)
                    elif feature_name in dataloader.dataset.categoricals:
                        feat_pos = dataloader.dataset.categoricals.index(feature_name)
                        if 'encoder_cat' in x_permuted:
                            perm_indices = torch.randperm(x_permuted['encoder_cat'].shape[0])
                            x_permuted['encoder_cat'][:, :, feat_pos] = x_permuted['encoder_cat'][perm_indices, :, feat_pos]
                        if 'decoder_cat' in x_permuted:
                            perm_indices = torch.randperm(x_permuted['decoder_cat'].shape[0])
                            x_permuted['decoder_cat'][:, :, feat_pos] = x_permuted['decoder_cat'][perm_indices, :, feat_pos]

                    # FIX: Access the prediction tensor
                    pred_permuted_output = model(x_permuted)
                    pred_permuted_tensor = pred_permuted_output.prediction
                    permuted_preds.append(pred_permuted_tensor.cpu().numpy())
            
            permuted_preds = np.concatenate(permuted_preds, axis=0)
            permuted_mae = mean_absolute_error(true_values.flatten(), permuted_preds.flatten())
            importance_values.append(permuted_mae - baseline_mae)
        
        importance_scores[feature_name] = np.mean(importance_values)
        progress_bar.progress((feat_idx + 1) / len(features_to_check))
    
    status_text.text("محاسبه اهمیت متغیرها تکمیل شد! ✅")
    return importance_scores

def calculate_simple_attention_weights(model, dataloader):
    """Extract attention weights from the model if available"""
    model.eval()
    attention_weights = []
    
    with torch.no_grad():
        for x, _ in dataloader:
            # FIX: Use tft.predict with return_attention=True for consistency
            try:
                raw_output = model.predict(x, mode="raw", return_attention=True)
                # The attention weights are in raw_output.attention
                attention_weights.append(raw_output.attention.cpu().numpy())
            except Exception as e:
                st.warning(f"Unable to extract attention weights. Error: {e}")
                return None
    
    if attention_weights:
        # The attention weights are typically [batch_size, num_heads, num_queries, num_keys]
        # We average over batches and heads
        return np.mean(np.concatenate(attention_weights, axis=0), axis=(0, 1))
    return None

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
        tft.eval() # Set model to evaluation mode after loading
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

        st.write("نمونهٔ خروجی پیش‌بینی")
        st.dataframe(pred_df.head(20))

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
            title=f"فروش تاریخی و پیش‌بینی‌شده برای store_id={sel}"
        ).interactive()

        st.altair_chart(overlay_chart, use_container_width=True)

    except Exception as e:
        st.error(f"خطا در پیش‌بینی: {e}")
        st.write("جزئیات خطا:", str(e))

# -----------------------------
# Enhanced Variable Importance Section
# -----------------------------
st.markdown("---")
st.subheader("🎯 اهمیت متغیرها (Feature Importance)")

# Create tabs for different importance methods
tab1, tab2, tab3 = st.tabs(["📋 لیست متغیرها", "🔄 Permutation Importance", "🧠 Attention Weights"])

with tab1:
    st.write("### متغیرهای ورودی مدل")
    
    # Continuous variables
    continuous_vars = training.reals if hasattr(training, 'reals') else []
    if continuous_vars:
        st.write("**متغیرهای پیوسته:**")
        for i, var in enumerate(continuous_vars, 1):
            st.write(f"{i}. `{var}`")
    
    # Categorical variables  
    categorical_vars = training.categoricals if hasattr(training, 'categoricals') else []
    if categorical_vars:
        st.write("**متغیرهای طبقه‌ای:**")
        for i, var in enumerate(categorical_vars, 1):
            st.write(f"{i}. `{var}`")
    else:
        st.write("**متغیرهای طبقه‌ای:** هیچ متغیر طبقه‌ای تعریف نشده")
    
    # Time-varying variables
    time_vars = training.time_varying_known_reals if hasattr(training, 'time_varying_known_reals') else []
    if time_vars:
        st.write("**متغیرهای متغیر با زمان:**")
        for i, var in enumerate(time_vars, 1):
            st.write(f"{i}. `{var}`")

with tab2:
    st.write("### محاسبه Permutation Importance")
    st.info("این روش اهمیت هر متغیر را با اندازه‌گیری تغییر عملکرد مدل پس از تصادفی کردن آن متغیر محاسبه می‌کند.")
    
    if st.button("🚀 محاسبه Permutation Importance", type="primary"):
        if 'dl' in locals() and dl is not None:
            try:
                importance_scores = calculate_permutation_importance(tft, dl, n_repeats=3)
                
                if importance_scores:
                    # Create DataFrame for plotting
                    importance_df = pd.DataFrame([
                        {"feature": k, "importance": v} for k, v in importance_scores.items()
                    ]).sort_values("importance", ascending=False)
                    
                    # Streamlit bar chart
                    st.write("**نمودار اهمیت متغیرها:**")
                    st.bar_chart(importance_df.set_index("feature")["importance"])
                    
                    # Show table
                    st.write("**جدول اهمیت متغیرها:**")
                    importance_df["importance"] = importance_df["importance"].round(4)
                    importance_df["رتبه"] = range(1, len(importance_df) + 1)
                    st.dataframe(importance_df[["رتبه", "feature", "importance"]], use_container_width=True)
                    
                    # Interpretation
                    st.write("**تفسیر نتایج:**")
                    top_feature = importance_df.iloc[0]
                    st.success(f"🏆 مهم‌ترین متغیر: `{top_feature['feature']}` با امتیاز {top_feature['importance']:.4f}")
                    
                    positive_features = importance_df[importance_df["importance"] > 0]
                    if len(positive_features) > 0:
                        st.info(f"📈 تعداد متغیرهای مؤثر (مثبت): {len(positive_features)}")
                    
                    negative_features = importance_df[importance_df["importance"] < 0]  
                    if len(negative_features) > 0:
                        st.warning(f"📉 متغیرهای با تأثیر منفی: {len(negative_features)}")
                    
                else:
                    st.warning("امکان محاسبه اهمیت متغیرها وجود ندارد.")
                    
            except Exception as e:
                st.error(f"خطا در محاسبه Permutation Importance: {e}")
        else:
            st.warning("ابتدا یک فروشگاه را انتخاب کنید و پیش‌بینی را اجرا کنید.")

with tab3:
    st.write("### وزن‌های Attention")
    st.info("این بخش وزن‌های attention مدل Temporal Fusion Transformer را نمایش می‌دهد.")
    
    if st.button("🔍 استخراج Attention Weights", type="primary"):
        if 'dl' in locals() and dl is not None:
            try:
                attention_weights = calculate_simple_attention_weights(tft, dl)
                
                if attention_weights is not None:
                    # Display attention weights
                    st.write("**وزن‌های Attention:**")
                    
                    # Create a simple visualization
                    # The attention weights are a 1D array of size num_encoder_steps
                    attention_df = pd.DataFrame({
                        "step": range(len(attention_weights)),
                        "attention_weight": attention_weights
                    })
                    
                    # Streamlit bar chart
                    st.bar_chart(attention_df.set_index("step")["attention_weight"])
                    
                    # Show table
                    st.write("**جدول وزن‌های Attention:**")
                    attention_df["attention_weight"] = attention_df["attention_weight"].round(4)
                    attention_df["رتبه"] = range(1, len(attention_df) + 1)
                    st.dataframe(attention_df[["رتبه", "step", "attention_weight"]], use_container_width=True)
                    
                else:
                    st.warning("امکان استخراج وزن‌های attention وجود ندارد.")
                    
            except Exception as e:
                st.error(f"خطا در استخراج Attention Weights: {e}")
        else:
            st.warning("ابتدا یک فروشگاه را انتخاب کنید و پیش‌بینی را اجرا کنید.")

# Model Info
st.markdown("---")
st.subheader("ℹ️ اطلاعات مدل")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("تعداد متغیرهای پیوسته", len(training.reals) if hasattr(training, 'reals') else 0)

with col2:
    st.metric("تعداد متغیرهای طبقه‌ای", len(training.categoricals) if hasattr(training, 'categoricals') else 0)

with col3:
    st.metric("طول پنجره پیش‌بینی", training.max_prediction_length if hasattr(training, 'max_prediction_length') else "نامشخص")

st.success("✅ اپلیکیشن با موفقیت بارگذاری شد!")

