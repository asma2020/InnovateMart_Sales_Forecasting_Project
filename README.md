# InnovateMart — Sales Forecasting Demo

A comprehensive sales forecasting system using Temporal Fusion Transformer (TFT) with PyTorch Forecasting. This project demonstrates time series forecasting on simulated retail sales data with an interactive Streamlit dashboard.

## 🎯 Project Overview

InnovateMart is a sales forecasting demonstration that combines:
- **Realistic sales data simulation** with seasonal patterns, promotions, and market shocks
- **State-of-the-art forecasting** using Temporal Fusion Transformer
- **Interactive visualization** with Streamlit dashboard
- **Multi-store analysis** across different store sizes and markets

## 🏗️ Repository Structure

```
innovatemart-forecast/
├── data/
│   └── simulated_sales.csv          # Generated sales data
├── models/                          # Trained model artifacts (created after training)
│   ├── tft_ckpt.pth                # Model weights
│   ├── training_dataset.pkl        # Training dataset object
│   ├── validation_dataset.pkl      # Validation dataset object
│   └── validation_raw.pkl          # Raw validation DataFrame
├── predictions/                     # Prediction outputs (optional)
│   ├── val_preds.npy               # Numpy predictions
│   ├── val_actuals.npy             # Actual values
│   └── val_timeidx.npy             # Time indices
├── simulate_data.py                 # Data simulation script
├── train_tft.py                     # Model training pipeline
├── predict_and_save.py             # Generate and save predictions
├── streamlit_app.py                 # Interactive dashboard
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
###  Clone the Repository
```bash
git clone https://github.com/asma2020/InnovateMart_Sales_Forecasting_Project.git
cd InnovateMart_Sales_Forecasting_Project
```


# Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data
```bash
python simulate_data.py
```
This creates `data/simulated_sales.csv` with 2 years of daily sales data for 4 stores.

### 3. Train Model
```bash
python train_tft.py
```
Trains the Temporal Fusion Transformer and saves model artifacts to `models/` directory.

### 4. Generate Predictions (Optional)
```bash
python predict_and_save.py
```
Creates standalone prediction files in `predictions/` directory.

### 5. Launch Dashboard
```bash
streamlit run streamlit_app.py
```
Open your browser to view the interactive forecasting dashboard.

## 📊 Features

### Data Simulation (`simulate_data.py`)
- **Multi-store environment**: 4 stores with different sizes (small, medium, large)
- **Realistic patterns**:
  - Seasonal trends (weekly, monthly, yearly)
  - Promotional campaigns and weekend effects
  - Holiday spikes and competitor impacts
  - Growth trends with realistic noise
- **Rich feature set**: Store characteristics, temporal features, promotional indicators

### Model Training (`train_tft.py`)
- **Temporal Fusion Transformer**: Advanced attention-based forecasting model
- **Feature engineering**: Automated lag features and moving averages
- **Robust preprocessing**: Handles missing values and categorical encoding
- **GPU support**: Automatic GPU detection and usage
- **Early stopping**: Prevents overfitting with validation monitoring

### Interactive Dashboard (`streamlit_app.py`)
- **Multi-language support**: Persian/English interface
- **Store-level analysis**: Individual store performance and forecasting
- **Visual comparisons**: Historical vs predicted sales with multiple horizons
- **Variable importance**: Model interpretability insights
- **Real-time forecasting**: Interactive prediction generation

## 🛠️ Technical Details

### Model Architecture
- **Temporal Fusion Transformer (TFT)**: Combines LSTM encoders with multi-head attention
- **Input features**:
  - Static: store_id, store_size, city_population
  - Time-varying known: promotions, temporal features
  - Time-varying unknown: sales, lags, moving averages
- **Forecast horizon**: 7 days ahead prediction
- **Quantile loss**: Provides prediction intervals

### Data Features
| Feature | Type | Description |
|---------|------|-------------|
| daily_sales | Target | Daily sales amount |
| promotion_active | Known | Promotional campaign indicator |
| day_of_week | Known | Day of week (0-6) |
| month | Known | Month of year |
| is_weekend | Known | Weekend indicator |
| store_size | Static | Store category (small/medium/large) |
| city_population | Static | Market size indicator |
| sales_lag_7/30 | Unknown | Lagged sales features |
| sales_ma_7/30 | Unknown | Moving average features |

## 📈 Model Performance

The TFT model includes several performance optimizations:
- **Attention mechanism**: Identifies important time steps and features
- **Variable selection**: Automatic feature importance calculation
- **Multi-horizon**: Simultaneous prediction for multiple future periods
- **Uncertainty quantification**: Prediction intervals via quantile regression

## 🎨 Dashboard Features

- **Data Loading & Model Artifacts** – Automatically loads preprocessed datasets and the trained TFT model, with caching for speed.
- **Store Selection** – Choose a `store_id` to view historical sales and predictions.
- **Historical Sales Chart** – Interactive Altair line chart showing daily sales trends for the selected store.
- **Forecast Visualization** – Model predictions overlaid with actual sales, broken down by forecast horizon.
- **Sample Predictions Table** – View raw forecast output including prediction horizon and corresponding dates.
- **Variable Importance** – Bar chart showing which features most influenced the model’s forecasts.
- **Error Handling** – Clear user messages if data is missing or calculations fail.


### Forecasting Interface
- Real-time prediction generation
- Multi-horizon forecast visualization
- Confidence intervals and uncertainty bands
- Model interpretability insights

### Interactive Controls
- Store selection dropdown
- Date range filtering
- Feature importance analysis
- Prediction export capabilities

## 📋 Requirements

Key dependencies include:
```
streamlit>=1.28.0
pytorch-forecasting>=1.0.0
pytorch-lightning>=2.0.0
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
altair>=4.2.0
scikit-learn>=1.3.0
```

See `requirements.txt` for complete dependency list.

## 🔧 Configuration

### Model Hyperparameters
- **Hidden size**: 16 (adjustable for model complexity)
- **Attention heads**: 1 (can increase for larger datasets)
- **Learning rate**: 1e-3 with plateau reduction
- **Batch size**: 64 (memory dependent)
- **Max epochs**: 8 with early stopping

### Data Parameters
- **Encoder length**: 30 days (historical context)
- **Prediction length**: 7 days (forecast horizon)
- **Validation split**: Last 37 days for testing

## 🚨 Troubleshooting

### Common Issues

1. **Missing model files**: Run `train_tft.py` before launching dashboard
2. **GPU memory errors**: Reduce batch_size in training script
3. **Import errors**: Ensure all requirements are installed
4. **Data format issues**: Check date parsing in data loading functions

### Performance Tips

- **GPU acceleration**: Ensure CUDA is available for faster training
- **Memory optimization**: Reduce sequence lengths for large datasets
- **Feature selection**: Remove irrelevant features to improve performance

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🔮 Future Enhancements

- [ ] Real-time data integration
- [ ] Additional forecasting models (Prophet, ARIMA)
- [ ] Advanced feature engineering pipeline
- [ ] Model ensemble capabilities
- [ ] Production deployment configuration
- [ ] A/B testing framework for promotional strategies

## 📞 Support

For questions or issues:
1. Check existing GitHub issues
2. Review troubleshooting section
3. Create new issue with detailed description

---



**Happy Forecasting! 📊✨**


