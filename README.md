# InnovateMart — Sales forecasting demo

## Overview
Simulation + TemporalFusionTransformer demo using pytorch-forecasting.

innovatemart-forecast/
├─ data/
│  └─ simulated_sales.csv            
├─ simulate_data.py                  
├─ train_tft.py                      
├─ predict_and_save.py               
├─ streamlit_app.py                  
├─ requirements.txt
└─ README.md

Files:
- simulate_data.py
- train_tft.py
- predict_and_save.py
- streamlit_app.py
- requirements.txt

## Quickstart
1. Create venv and install requirements:
   pip install -r requirements.txt
2. Simulate data:
   python simulate_data.py
3. Train model:
   python train_tft.py
4. (optional) Produce predictions:
   python predict_and_save.py
5. Run Streamlit:
   streamlit run streamlit_app.py


