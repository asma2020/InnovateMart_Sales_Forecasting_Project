import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import altair as alt
from pytorch_forecasting import TemporalFusionTransformer
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import copy

# -----------------------------
# App config
# -----------------------------
st.set_page_config(layout="wide", page_title="InnovateMart Forecast Viewer")
st.title("InnovateMart — مشاهدهٔ پیش‌بینی‌های TFT")

# -----------------------------
# I/O utilities
# -----------------------------
@st.cache_resource
def load_pickles(model_dir="models"):
    paths = {
        "training_ds": os.path.join(model_dir, "training_dataset.pkl"),
        "validation_ds": os.path.join(model_dir, "validation_dataset.pkl"),
        "validation_raw": os.path.join(model_dir, "validation_raw.pkl"),
        "ckpt": os.path.join(model_dir, "tft_ckpt.pth"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            st.error(f"فایل پیدا نشد: {p}")
    with open(paths["training_ds"], "rb") as f:
        training = pickle.load(f)
    with open(paths["validation_ds"], "rb") as f:
        validation = pickle.load(f)
    with open(paths["validation_raw"], "rb") as f:
        val_raw = pickle.load(f)
    ckpt_path = paths["ckpt"]
    return training, validation, val_raw, ckpt_path

# -----------------------------
# Model build / load
# -----------------------------
from pytorch_forecasting.metrics import QuantileLoss

@st.cache_resource
def build_and_load_model(_training_ds, ckpt_path, device="cpu"):
    tft = TemporalFusionTransformer.from_dataset(
        _training_ds,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=1,
        dropout=0.1,
        hidden_continuous_size=8,
        log_interval=10,
        reduce_on_plateau_patience=2,
        loss=QuantileLoss(quantiles=[0.5]),
    )
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    tft.load_state_dict(state)
    tft.to(device)
    tft.eval()
    return tft

# -----------------------------
# Prediction helpers
# -----------------------------

def predict_on_validation(tft, validation_ds, batch_size=64, device="cpu"):
    val_dl = validation_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    preds = tft.predict(val_dl)
    return preds


def map_preds_to_dates_fixed(preds, validation_raw, validation_ds):
    """
    نگاشت پیش‌بینی‌ها به فروشگاه‌ها و تاریخ‌ها (برای همه فروشگاه‌ها)
    """
    rows = []
    stores = validation_raw["store_id"].unique()
    horizon = preds.shape[1]

    for i, store in enumerate(stores):
        store_min_time = validation_raw.loc[validation_raw["store_id"] == store, "time_idx"].max() - horizon + 1
        for h in range(horizon):
            time_idx = store_min_time + h
            rows.append({
                "store_id": store,
                "time_idx": time_idx,
                "pred": float(preds[i, h]),
                "horizon_step": h
            })

    pred_df = pd.DataFrame(rows)
    time_to_date = validation_raw[["time_idx", "date"]].drop_duplicates()
    pred_df = pred_df.merge(time_to_date, on="time_idx", how="left")

    actuals = validation_raw[["store_id", "time_idx", "date", "daily_sales"]].drop_duplicates()
    merged = pred_df.merge(actuals, on=["store_id", "time_idx"], how="left", suffixes=("", "_actual"))
    merged["date"] = merged["date_actual"].fillna(merged["date"])
    merged = merged.drop(columns=["date_actual"], errors="ignore")
    return merged

# -----------------------------
# Plotting utilities
# -----------------------------

def plot_store_comparison(merged_df, store=None, aggregate=False):
    if aggregate:
        df_plot = merged_df.groupby("date").agg(
            actual=("daily_sales", "sum"),
            pred=("pred", "sum")
        ).reset_index()
        title_suffix = "(تجمیعی - همه فروشگاه‌ها)"
    else:
        if store is None:
            available_stores = merged_df["store_id"].unique()
            if len(available_stores) > 0:
                store = available_stores[0]
            else:
                st.warning("هیچ فروشگاهی در داده‌ها یافت نشد!")
                return

        store_data = merged_df[merged_df["store_id"] == store]
        if store_data.empty:
            st.warning(f"داده‌ای برای فروشگاه {store} یافت نشد!")
            return

        df_plot = store_data.groupby("date").agg(
            actual=("daily_sales", "mean"),
            pred=("pred", "mean")
        ).reset_index()
        title_suffix = f"(فروشگاه {store})"

    if df_plot.empty:
        st.warning(f"داده‌ای برای نمایش یافت نشد {title_suffix}")
        return

    has_actual = not df_plot['actual'].isna().all()

    if has_actual:
        df_long = df_plot.melt(id_vars=["date"], value_vars=["actual", "pred"],
                              var_name="kind", value_name="sales")
        color_scale = alt.Scale(domain=['actual', 'pred'], range=['#1f77b4', '#ff7f0e'])
        chart = alt.Chart(df_long).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("date:T", title="تاریخ"),
            y=alt.Y("sales:Q", title="فروش"),
            color=alt.Color("kind:N", title="نوع", scale=color_scale),
            tooltip=["date:T", "kind:N", "sales:Q"]
        ).properties(width=900, height=400, title=f"مقایسه فروش واقعی و پیش‌بینی {title_suffix}")
    else:
        chart = alt.Chart(df_plot).mark_line(point=True, strokeWidth=2).encode(
            x=alt.X("date:T", title="تاریخ"),
            y=alt.Y("pred:Q", title="فروش پیش‌بینی شده"),
            tooltip=["date:T", "pred:Q"]
        ).properties(width=900, height=400, title=f"پیش‌بینی فروش {title_suffix}")

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Permutation feature importance (robust)
# -----------------------------

def permute_feature_in_batch(x, feature_name, encoder_vars, decoder_vars):
    """
    permute کردن یک ویژگی در x (فقط x، نه کل batch).
    خروجی همان نوع ورودی را دارد.
    """
    if x is None:
        return x

    def _shuffle_along_batch(tensor):
        if not torch.is_tensor(tensor) or tensor.shape[0] <= 1:
            return tensor
        idx = torch.randperm(tensor.shape[0], device=tensor.device)
        return tensor[idx].clone()

    # dict (رایج‌ترین حالت برای pytorch-forecasting)
    if isinstance(x, dict):
        x_copy = {}
        if feature_name in encoder_vars:
            feature_idx = encoder_vars.index(feature_name)
            section = "encoder"
        elif feature_name in decoder_vars:
            feature_idx = decoder_vars.index(feature_name)
            section = "decoder"
        else:
            return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}

        # continuous
        if section == "encoder" and "encoder_cont" in x and torch.is_tensor(x["encoder_cont"]):
            ec = x["encoder_cont"].clone()
            if ec.ndim >= 3 and feature_idx < ec.shape[-1]:
                vals = ec[:, :, feature_idx]
                perm_idx = torch.randperm(vals.shape[0], device=vals.device)
                ec[:, :, feature_idx] = vals[perm_idx]
                x_copy.update(x)
                x_copy["encoder_cont"] = ec
                return x_copy

        if section == "decoder" and "decoder_cont" in x and torch.is_tensor(x["decoder_cont"]):
            dc = x["decoder_cont"].clone()
            if dc.ndim >= 3 and feature_idx < dc.shape[-1]:
                vals = dc[:, :, feature_idx]
                perm_idx = torch.randperm(vals.shape[0], device=vals.device)
                dc[:, :, feature_idx] = vals[perm_idx]
                x_copy.update(x)
                x_copy["decoder_cont"] = dc
                return x_copy

        # categorical
        if section == "encoder" and "encoder_cat" in x and torch.is_tensor(x["encoder_cat"]):
            ec = x["encoder_cat"].clone()
            if ec.ndim >= 3 and feature_idx < ec.shape[-1]:
                vals = ec[:, :, feature_idx]
                perm_idx = torch.randperm(vals.shape[0], device=vals.device)
                ec[:, :, feature_idx] = vals[perm_idx]
                x_copy.update(x)
                x_copy["encoder_cat"] = ec
                return x_copy

        if section == "decoder" and "decoder_cat" in x and torch.is_tensor(x["decoder_cat"]):
            dc = x["decoder_cat"].clone()
            if dc.ndim >= 3 and feature_idx < dc.shape[-1]:
                vals = dc[:, :, feature_idx]
                perm_idx = torch.randperm(vals.shape[0], device=vals.device)
                dc[:, :, feature_idx] = vals[perm_idx]
                x_copy.update(x)
                x_copy["decoder_cat"] = dc
                return x_copy

        # fallback تلاش برای permute در هر tensor که آخرین بعدش >= feature_idx
        x_copy.update(x)
        for k, v in x.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[-1] > feature_idx:
                try:
                    v_new = v.clone()
                    vals = v_new[..., feature_idx]
                    perm_idx = torch.randperm(vals.shape[0], device=vals.device)
                    v_new[..., feature_idx] = vals[perm_idx]
                    x_copy[k] = v_new
                    return x_copy
                except Exception:
                    continue

        return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in x.items()}

    # tuple (مثلاً (encoder, decoder)) — پردازش هر قسمت
    if isinstance(x, tuple):
        out_parts = []
        for part in x:
            if isinstance(part, dict):
                out_parts.append(permute_feature_in_batch(part, feature_name, encoder_vars, decoder_vars))
            elif torch.is_tensor(part):
                out_parts.append(_shuffle_along_batch(part))
            else:
                out_parts.append(part)
        return tuple(out_parts)

    # tensor ساده
    if torch.is_tensor(x):
        return _shuffle_along_batch(x)

    return x


def calculate_permutation_importance(tft, validation_ds, device="cpu", n_repeats=3):
    """
    محاسبه اهمیت ویژگی‌ها با روش permutation (نسخه فیکس شده)
    """
    val_dl = validation_ds.to_dataloader(train=False, batch_size=32, num_workers=0)

    # baseline predictions
    baseline_predictions = tft.predict(val_dl)

    # دریافت actual values
    actuals = []
    for batch in val_dl:
        if isinstance(batch, tuple) and len(batch) >= 2:
            y = batch[1]
            if isinstance(y, tuple):
                y = y[0]
            if torch.is_tensor(y):
                actuals.append(y.detach().cpu().numpy())
        elif hasattr(batch, "y") and torch.is_tensor(batch.y):
            actuals.append(batch.y.detach().cpu().numpy())

    if actuals:
        actuals = np.concatenate(actuals, axis=0)
        baseline_mse = mean_squared_error(actuals.flatten(), baseline_predictions.flatten())
    else:
        baseline_mse = np.var(baseline_predictions.flatten())
        st.info("⚠️ actual values پیدا نشد — از variance predictions استفاده شد.")

    encoder_vars = getattr(tft, "encoder_variables", []) or []
    decoder_vars = getattr(tft, "decoder_variables", []) or []
    all_features = list(dict.fromkeys(encoder_vars + decoder_vars))

    importance_scores = {}
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, feature in enumerate(all_features):
        status_text.text(f"در حال محاسبه اهمیت {feature}...")
        feature_scores = []

        for repeat in range(n_repeats):
            val_dl_permuted = validation_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
            permuted_predictions = []

            for batch in val_dl_permuted:
                if isinstance(batch, tuple) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch, None

                # فقط x رو permute می‌کنیم
                x_permuted = permute_feature_in_batch(x, feature, encoder_vars, decoder_vars)

                with torch.no_grad():
                    # مدل همیشه انتظار dict داره — اگر tuple شد، اولین عضو dict را امتحان کن
                    if isinstance(x_permuted, tuple):
                        x_input = x_permuted[0] if isinstance(x_permuted[0], dict) else x_permuted
                    else:
                        x_input = x_permuted

                    pred = tft(x_input)

                    if isinstance(pred, dict):
                        pred = pred.get("prediction", next(iter(pred.values())))
                    if isinstance(pred, (list, tuple)):
                        pred = pred[0]
                    if torch.is_tensor(pred):
                        permuted_predictions.append(pred.detach().cpu().numpy())
                    else:
                        permuted_predictions.append(np.asarray(pred))

            if len(permuted_predictions) == 0:
                continue

            permuted_predictions = np.concatenate(permuted_predictions, axis=0)

            if actuals is not None and len(actuals) > 0:
                permuted_mse = mean_squared_error(actuals.flatten(), permuted_predictions.flatten())
            else:
                permuted_mse = np.var(permuted_predictions.flatten())

            feature_scores.append(permuted_mse - baseline_mse)

        importance_scores[feature] = float(np.mean(feature_scores)) if feature_scores else 0.0
        progress_bar.progress((idx + 1) / max(1, len(all_features)))

    progress_bar.empty()
    status_text.empty()
    return importance_scores

# -----------------------------
# Session state initialization
# -----------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.available_stores = []

# -----------------------------
# Sidebar / UI controls
# -----------------------------
st.sidebar.header("تنظیمات")
model_dir = st.sidebar.text_input("مسیر پوشهٔ مدل", value="models")
device_opt = st.sidebar.selectbox("Device", ["cpu", "cuda" if torch.cuda.is_available() else "cpu"])
batch_size = st.sidebar.slider("Batch size برای پیش‌بینی", min_value=16, max_value=256, value=64, step=16)

display_mode = st.sidebar.radio(
    "نوع نمایش",
    ["تجمیعی (همه فروشگاه‌ها)", "فروشگاه خاص"],
    index=0
)

aggregate = display_mode == "تجمیعی (همه فروشگاه‌ها)"

selected_store = None
if not aggregate and st.session_state.data_loaded:
    selected_store = st.sidebar.selectbox(
        "انتخاب فروشگاه",
        options=st.session_state.available_stores,
        index=0 if st.session_state.available_stores else None
    )

# -----------------------------
# Main flow: load, predict, show
# -----------------------------
if st.button("بارگذاری مدل و اجرا"):
    with st.spinner("در حال بارگذاری فایل‌ها و مدل..."):
        try:
            training_ds, validation_ds, validation_raw, ckpt_path = load_pickles(model_dir)

            # debug info
            st.write("### اطلاعات دیباگ")
            st.write(f"تعداد رکوردهای validation_raw: {len(validation_raw)}")
            st.write(f"فروشگاه‌های موجود: {validation_raw['store_id'].unique()}")
            st.write(f"بازه time_idx: {validation_raw['time_idx'].min()} تا {validation_raw['time_idx'].max()}")
            st.write(f"بازه تاریخ: {validation_raw['date'].min()} تا {validation_raw['date'].max()}")

            st.session_state.available_stores = sorted(validation_raw["store_id"].unique().tolist())
            st.session_state.data_loaded = True

            tft = build_and_load_model(training_ds, ckpt_path, device=device_opt)
            st.success("مدل بارگذاری شد.")

            preds = predict_on_validation(tft, validation_ds, batch_size=batch_size, device=device_opt)
            st.write("شکل پیش‌بینی‌ها:", preds.shape)

            merged = map_preds_to_dates_fixed(preds, validation_raw, validation_ds)

            if merged.empty:
                st.error("نتیجهٔ نگاشت خالی شد!")
            else:
                st.write(f"تعداد رکوردهای نگاشت یافته: {len(merged)}")
                st.write("نمونه از داده‌های نگاشت یافته:")
                st.dataframe(merged.head(10))

                st.session_state.merged_data = merged
                st.session_state.tft_model = tft
                st.session_state.validation_ds = validation_ds

                # metrics + plots
                if aggregate:
                    df_eval = merged.groupby("date").agg(actual=("daily_sales", "sum"), pred=("pred", "sum")).dropna(subset=['actual'])
                    if not df_eval.empty:
                        y_true = df_eval["actual"].values
                        y_pred = df_eval["pred"].values
                        mae_val = mean_absolute_error(y_true, y_pred)
                        rmse_val = math.sqrt(mean_squared_error(y_true, y_pred))
                        col1, col2 = st.columns(2)
                        col1.metric("MAE (تجمیعی)", f"{mae_val:.3f}")
                        col2.metric("RMSE (تجمیعی)", f"{rmse_val:.3f}")
                    else:
                        st.info("داده واقعی برای محاسبه متریک‌ها موجود نیست.")
                    st.markdown("### نمودار واقعی vs پیش‌بینی (تجمیعی)")
                    plot_store_comparison(merged, aggregate=True)
                else:
                    if selected_store is None and st.session_state.available_stores:
                        selected_store = st.session_state.available_stores[0]
                        st.info(f"فروشگاه {selected_store} به طور خودکار انتخاب شد.")

                    if selected_store is not None:
                        st.markdown(f"### نتایج فروشگاه {selected_store}")
                        store_data = merged[merged["store_id"] == selected_store]
                        if store_data.empty:
                            st.error(f"هیچ داده‌ای برای فروشگاه {selected_store} یافت نشد!")
                        else:
                            st.write(f"تعداد رکوردهای فروشگاه {selected_store}: {len(store_data)}")
                            df_eval = store_data.dropna(subset=['daily_sales'])
                            if not df_eval.empty:
                                y_true = df_eval["daily_sales"].values
                                y_pred = df_eval["pred"].values
                                mae_val = mean_absolute_error(y_true, y_pred)
                                rmse_val = math.sqrt(mean_squared_error(y_true, y_pred))
                                col1, col2 = st.columns(2)
                                col1.metric(f"MAE (فروشگاه {selected_store})", f"{mae_val:.3f}")
                                col2.metric(f"RMSE (فروشگاه {selected_store})", f"{rmse_val:.3f}")
                            else:
                                st.info(f"داده واقعی برای فروشگاه {selected_store} موجود نیست.")

                            plot_store_comparison(merged, store=selected_store, aggregate=False)
                    else:
                        st.warning("هیچ فروشگاهی برای انتخاب موجود نیست.")

                # summary
                st.markdown("### آمار کلی")
                col1, col2, col3 = st.columns(3)
                col1.metric("تعداد فروشگاه‌ها", len(merged["store_id"].unique()))
                col2.metric("تعداد روزهای پیش‌بینی", len(merged["date"].unique()))
                col3.metric("تعداد کل رکوردهای پیش‌بینی", len(merged))

                st.markdown("### خلاصه پیش‌بینی‌ها برای هر فروشگاه")
                summary = merged.groupby('store_id').agg({
                    'pred': ['count', 'mean', 'std'],
                    'daily_sales': ['count', 'mean', 'std']
                }).round(2)
                summary.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in summary.columns]
                st.dataframe(summary)

        except Exception as e:
            st.error(f"خطا در اجرای برنامه: {str(e)}")
            st.exception(e)

# -----------------------------
# Feature importance UI
# -----------------------------
if st.session_state.data_loaded:
    st.markdown("---")
    st.markdown("## اهمیت ویژگی‌ها")
    tab1, tab2 = st.tabs(["روش کلاسیک TFT", "روش Permutation"])

    with tab1:
        st.markdown("### روش کلاسیک TFT (interpret_output)")
        if st.button("محاسبه اهمیت ویژگی‌ها (کلاسیک)"):
            with st.spinner("در حال محاسبه اهمیت ویژگی‌ها..."):
                try:
                    training_ds, validation_ds, validation_raw, ckpt_path = load_pickles(model_dir)
                    tft = build_and_load_model(training_ds, ckpt_path, device=device_opt)
                    val_dl = validation_ds.to_dataloader(train=False, batch_size=min(batch_size, 32), num_workers=0)
                    st.info("در حال محاسبه raw predictions...")
                    raw_predictions, x, *_ = tft.predict(val_dl, mode="raw", return_x=True)
                    st.info("در حال تفسیر نتایج...")
                    interpretation = tft.interpret_output(raw_predictions, reduction="mean")
                    st.success("محاسبه اهمیت ویژگی‌ها کامل شد!")
                    st.write("**کلیدهای موجود در interpretation:**")
                    st.write(list(interpretation.keys()))
                    if "attention" in interpretation:
                        st.subheader("نقشه توجه (Attention Map)")
                        attention = interpretation["attention"]
                        st.write(f"شکل attention: {attention.shape}")
                    if hasattr(tft, 'encoder_variables') and hasattr(tft, 'decoder_variables'):
                        st.subheader("ویژگی‌های مدل")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Encoder Variables:**")
                            st.write(tft.encoder_variables)
                        with col2:
                            st.write("**Decoder Variables:**")
                            st.write(tft.decoder_variables)
                    available_keys = [k for k in interpretation.keys() if 'importance' in k.lower()]
                    if not available_keys:
                        st.warning("هیچ اطلاعات اهمیت ویژگی یافت نشد در روش کلاسیک.")
                        st.info("لطفا از روش Permutation استفاده کنید.")
                except Exception as e:
                    st.error(f"خطا در محاسبه اهمیت ویژگی‌ها: {str(e)}")
                    st.exception(e)

    with tab2:
        st.markdown("### روش Permutation Importance")
        col1, col2 = st.columns(2)
        with col1:
            n_repeats = st.slider("تعداد تکرار برای هر ویژگی", min_value=1, max_value=10, value=3, help="تعداد بیشتر = نتیجه دقیق‌تر اما زمان بیشتر")
        with col2:
            use_sample = st.checkbox("استفاده از نمونه کوچک", value=True, help="برای سرعت بیشتر، فقط قسمتی از داده‌ها استفاده شود")

        if st.button("محاسبه اهمیت ویژگی‌ها (Permutation Method)"):
            with st.spinner("در حال محاسبه اهمیت ویژگی‌ها با روش Permutation..."):
                try:
                    training_ds, validation_ds, validation_raw, ckpt_path = load_pickles(model_dir)
                    tft = build_and_load_model(training_ds, ckpt_path, device=device_opt)
                    if use_sample:
                        st.info(f"استفاده از batch size کوچک برای تسریع محاسبات...")
                    importance_scores = calculate_permutation_importance(tft, validation_ds, device=device_opt, n_repeats=n_repeats)
                    if importance_scores:
                        df_importance = pd.DataFrame([{"feature": feature, "importance": score} for feature, score in importance_scores.items()]).sort_values("importance", ascending=False)
                        st.success("محاسبه اهمیت ویژگی‌ها با روش Permutation کامل شد!")
                        st.subheader("🎯 اهمیت ویژگی‌ها (Permutation Importance)")
                        st.dataframe(df_importance, use_container_width=True)
                        chart = alt.Chart(df_importance).mark_bar().encode(x=alt.X("importance:Q", title="اهمیت (افزایش MSE)"), y=alt.Y("feature:N", sort="-x", title="ویژگی"), color=alt.Color("importance:Q", scale=alt.Scale(scheme="viridis")), tooltip=["feature:N", "importance:Q"]).properties(width=800, height=400, title="اهمیت ویژگی‌ها بر اساس روش Permutation")
                        st.altair_chart(chart, use_container_width=True)
                        st.markdown("### 📊 تفسیر نتایج:")
                        top_features = df_importance.head(3)
                        st.markdown("**🏆 مهم‌ترین ویژگی‌ها:**")
                        for idx, row in top_features.iterrows():
                            feature = row['feature']
                            importance = row['importance']
                            if importance > 0:
                                st.markdown(f"- **{feature}**: افزایش {importance:.4f} در MSE (ویژگی مهم)")
                            else:
                                st.markdown(f"- **{feature}**: کاهش {abs(importance):.4f} در MSE (ممکن است noise باشد)")
                        with st.expander("💡 راهنمای تفسیر نتایج"):
                            st.markdown("""
                            **نحوه تفسیر Permutation Importance:**
                            - **مقدار مثبت**: با تغییر تصادفی این ویژگی، دقت مدل کاهش می‌یابد → ویژگی مهم است
                            - **مقدار منفی**: با تغییر تصادفی این ویژگی، دقت مدل بهتر می‌شود → ممکن است ویژگی noise یا غیرمهم باشد
                            - **مقدار نزدیک صفر**: ویژگی تاثیر چندانی بر مدل ندارد
                            **ویژگی‌های معمولاً مهم در پیش‌بینی فروش:**
                            - `daily_sales`: مقدار فروش قبلی (target)
                            - `sales_lag_7`, `sales_lag_30`: فروش با تاخیر
                            - `sales_ma_7`, `sales_ma_30`: میانگین متحرک فروش
                            - `day_of_week`: روز هفته
                            - `promotion_active`: وضعیت تخفیف
                            """)
                        st.download_button(label="📥 دانلود نتایج (CSV)", data=df_importance.to_csv(index=False).encode('utf-8'), file_name=f"feature_importance_permutation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")
                    else:
                        st.error("محاسبه اهمیت ویژگی‌ها انجام نشد!")
                except Exception as e:
                    st.error(f"خطا در محاسبه Permutation Importance: {str(e)}")
                    st.exception(e)
                    with st.expander("🔧 راهنمای رفع مشکل"):
                        st.markdown("""
                        **مشکلات احتمالی و راه حل:**
                        1. **کمبود حافظه**: 
                           - تعداد تکرار را کم کنید
                           - گزینه "استفاده از نمونه کوچک" را فعال کنید
                           - batch size را کاهش دهید
                        2. **مشکل در ساختار batch**:
                           - ممکن است نیاز به تطبیق کد با ساختار خاص مدل شما باشد
                           - بررسی کنید که آیا نام ویژگی‌ها درست است
                        3. **زمان زیاد محاسبه**:
                           - تعداد تکرار را به 1 یا 2 کاهش دهید
                           - از نمونه کوچک‌تر استفاده کنید
                        """)

st.success("پایان.")

# -----------------------------
# Usage help when not loaded
# -----------------------------
if not st.session_state.data_loaded:
    st.info("برای شروع، روی دکمه 'بارگذاری مدل و اجرا' کلیک کنید.")
    with st.expander("راهنمای استفاده"):
        st.markdown("""
        ### نحوه استفاده:
        1. مسیر پوشه مدل را در نوار کناری تنظیم کنید
        2. تنظیمات مورد نظر (device، batch size) را انتخاب کنید  
        3. نوع نمایش را انتخاب کنید:
           - **تجمیعی**: نمایش مجموع فروش همه فروشگاه‌ها
           - **فروشگاه خاص**: نمایش فروش یک فروشگاه خاص
        4. روی دکمه "بارگذاری مدل و اجرا" کلیک کنید
        5. برای محاسبه اهمیت ویژگی‌ها، از تب‌های "اهمیت ویژگی‌ها" استفاده کنید
        """)
        st.markdown("---")
        st.markdown("""
        ### درباره روش‌های Feature Importance:
        **1. روش کلاسیک TFT:**
        - از attention mechanism مدل استفاده می‌کند
        - نتایج سریع‌تر اما محدود به ساختار مدل
        - ممکن است همیشه کار نکند
        **2. روش Permutation:**
        - مستقل از نوع مدل
        - اندازه‌گیری تاثیر واقعی هر ویژگی
        - زمان‌بر اما دقیق‌تر و قابل‌اعتمادتر
        - با هر نوع مدل ML سازگار است
        """)
