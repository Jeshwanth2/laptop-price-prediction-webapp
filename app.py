import streamlit as st
import pandas as pd
import numpy as np
import joblib
from train_model import LaptopPricePredictor

# Load the trained model
@st.cache_resource
def load_predictor():
    predictor = LaptopPricePredictor()
    predictor.load_saved_model('laptop_price_regression_model.joblib')
    return predictor

predictor = load_predictor()

# For dropdowns, load the original data to get unique values
df = pd.read_csv('laptop_price.csv', encoding='latin1')

# Preprocess for dropdowns
df['Ram'] = df['Ram'].str.replace('GB', '', regex=False).astype(int)
if 'Weight' in df.columns:
    df['Weight'] = df['Weight'].str.replace('kg', '', regex=False).astype(float)
df['Inches'] = df['Inches'].astype(float)
df['Memory'] = df['Memory'].fillna('')

# Helper functions (copy from train_model.py for dropdowns)
def extract_memory_type(memory_str):
    if not isinstance(memory_str, str):
        return 'Other'
    if 'SSD' in memory_str:
        return 'SSD'
    elif 'HDD' in memory_str:
        return 'HDD'
    elif 'Flash Storage' in memory_str:
        return 'Flash'
    elif 'Hybrid' in memory_str:
        return 'Hybrid'
    else:
        return 'Other'

def extract_memory_size(memory_str):
    if not isinstance(memory_str, str):
        return 0
    import re
    sizes = re.findall(r'(\d+\.?\d*)\s*(GB|TB)', memory_str)
    total = 0.0
    for size, unit in sizes:
        try:
            if unit == 'TB':
                total += float(size) * 1024
            else:
                total += float(size)
        except ValueError:
            continue
    return round(total)

def extract_cpu_brand(cpu_str):
    return cpu_str.split()[0] if isinstance(cpu_str, str) else 'Unknown'

def extract_cpu_type(cpu_str):
    return ' '.join(cpu_str.split()[1:3]) if isinstance(cpu_str, str) else 'Unknown'

def extract_cpu_speed(cpu_str):
    import re
    if not isinstance(cpu_str, str):
        return 2.0
    speed_str = re.findall(r'\d\.\d+GHz', cpu_str)
    if speed_str:
        return float(speed_str[0].replace('GHz', ''))
    return 2.0

def extract_gpu_brand(gpu_str):
    return gpu_str.split()[0] if isinstance(gpu_str, str) else 'Unknown'

def extract_resolution(resolution_str):
    import re
    if not isinstance(resolution_str, str):
        return '1366x768'
    res = re.findall(r'(\d+x\d+)', resolution_str)
    return res[0] if res else '1366x768'

#engineered features for dropdowns
df['Memory_Type'] = df['Memory'].apply(extract_memory_type)
df['Memory_Size_GB'] = df['Memory'].apply(extract_memory_size)
df['CPU_Brand'] = df['Cpu'].apply(extract_cpu_brand)
df['CPU_Type'] = df['Cpu'].apply(extract_cpu_type)
df['CPU_Speed_GHz'] = df['Cpu'].apply(extract_cpu_speed)
df['GPU_Brand'] = df['Gpu'].apply(extract_gpu_brand)
df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen').fillna(0).astype(int)
df['IPS'] = df['ScreenResolution'].str.contains('IPS').fillna(0).astype(int)
resolution = df['ScreenResolution'].apply(extract_resolution)
df['Resolution_X'] = resolution.apply(lambda x: int(x.split('x')[0]) if x else 1366)
df['Resolution_Y'] = resolution.apply(lambda x: int(x.split('x')[1]) if x else 768)

# UI
st.markdown('<div class="title-box"><h1>💻 Laptop Price Predictor</h1></div>', unsafe_allow_html=True)
st.markdown("""
    <div class="subtitle-box">
    <p style="font-size:17px; color:#333;">Enter your laptop specifications to predict its price in euros.</p>
    </div>
""", unsafe_allow_html=True)

with st.form("laptop_specs"):
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox("Company", sorted(df['Company'].unique()))
        type_name = st.selectbox("Type Name", sorted(df['TypeName'].unique()))
        ram = st.selectbox("RAM (GB)", sorted(df['Ram'].unique()))
        weight = st.number_input("Weight (kg)", min_value=float(df['Weight'].min()), max_value=float(df['Weight'].max()), value=float(df['Weight'].median()))
        inches = st.slider("Screen Size (inches)", float(df['Inches'].min()), float(df['Inches'].max()), float(df['Inches'].median()), 0.1)
        memory_type = st.selectbox("Memory Type", sorted(df['Memory_Type'].unique()))
        memory_size = st.selectbox("Memory Size (GB)", sorted(df['Memory_Size_GB'].unique()))
    with col2:
        cpu_brand = st.selectbox("CPU Brand", sorted(df['CPU_Brand'].unique()))
        cpu_type = st.selectbox("CPU Type", sorted(df['CPU_Type'].unique()))
        cpu_speed = st.selectbox("CPU Speed (GHz)", sorted(df['CPU_Speed_GHz'].unique()))
        gpu_brand = st.selectbox("GPU Brand", sorted(df['GPU_Brand'].unique()))
        touchscreen = st.selectbox("Touchscreen", [0, 1])
        ips = st.selectbox("IPS", [0, 1])
        resolution_x = st.selectbox("Resolution X", sorted(df['Resolution_X'].unique()))
        resolution_y = st.selectbox("Resolution Y", sorted(df['Resolution_Y'].unique()))

    submit_button = st.form_submit_button("Predict Price (€)")

if submit_button:
    #  feature_columns
    input_features = {
        'Company': company,
        'TypeName': type_name,
        'Ram': int(ram),
        'Weight': float(weight),
        'Inches': float(inches),
        'Memory_Type': memory_type,
        'Memory_Size_GB': int(memory_size),
        'CPU_Brand': cpu_brand,
        'CPU_Type': cpu_type,
        'CPU_Speed_GHz': float(cpu_speed),
        'GPU_Brand': gpu_brand,
        'Touchscreen': int(touchscreen),
        'IPS': int(ips),
        'Resolution_X': int(resolution_x),
        'Resolution_Y': int(resolution_y)
    }
    try:
        price = predictor.predict_price(input_features)
        st.markdown(f'<div class="prediction-box">Predicted Price: <b>€{price}</b></div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.markdown("""
    <div class="footer-box">
    © 2025 Laptop Price Predictor | Linear Regression Model<br>
    Developed by <b>Jeshwanth Basutkar</b>
    </div>
""", unsafe_allow_html=True)