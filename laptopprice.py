import streamlit as st
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("💻 Laptop Price Predictor")


# ── TRAIN (cached – runs once) ────────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = pd.read_csv("laptop_data.csv")

    df.drop(columns=['Unnamed: 0'], inplace=True)

    df['Ram']    = df['Ram'].str.replace('GB', '').astype('int32')
    df['Weight'] = df['Weight'].str.replace('kg', '').astype('float32')

    df['Touchscreen'] = df['ScreenResolution'].apply(
        lambda x: 1 if 'Touchscreen' in x else 0)
    df['Ips'] = df['ScreenResolution'].apply(
        lambda x: 1 if 'IPS' in x else 0)

    new = df['ScreenResolution'].str.split('x', n=1, expand=True)
    df['X_res'] = new[0].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
    df['Y_res'] = new[1]
    df['X_res'] = df['X_res'].astype('int')
    df['Y_res'] = df['Y_res'].astype('int')
    df['ppi']   = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float')
    df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

    df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))

    def fetch_processor(text):
        if text in ('Intel Core i7', 'Intel Core i5', 'Intel Core i3'):
            return text
        return 'Other Intel Processor' if text.split()[0] == 'Intel' else 'AMD Processor'

    df['Cpu brand'] = df['Cpu Name'].apply(fetch_processor)
    df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

    # Memory parsing
    df['Memory'] = (df['Memory'].astype(str)
                    .str.replace(r'\.0', '', regex=True)
                    .str.replace('GB', '')
                    .str.replace('TB', '000'))

    new2 = df['Memory'].str.split('+', n=1, expand=True)
    df['first']  = new2[0].str.strip().astype(str)
    # fillna then force str so NaN never reaches the 'HDD' in x check
    df['second'] = new2[1].fillna('0').astype(str).str.strip()

    df['Layer1HDD']           = df['first'].apply(lambda x: 1 if 'HDD'           in x else 0)
    df['Layer1SSD']           = df['first'].apply(lambda x: 1 if 'SSD'           in x else 0)
    df['Layer1Hybrid']        = df['first'].apply(lambda x: 1 if 'Hybrid'        in x else 0)
    df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
    df['first'] = df['first'].str.replace(r'\D', '', regex=True)

    df['Layer2HDD']           = df['second'].apply(lambda x: 1 if 'HDD'           in x else 0)
    df['Layer2SSD']           = df['second'].apply(lambda x: 1 if 'SSD'           in x else 0)
    df['Layer2Hybrid']        = df['second'].apply(lambda x: 1 if 'Hybrid'        in x else 0)
    df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
    df['second'] = df['second'].str.replace(r'\D', '', regex=True)

    df['first']  = df['first'].astype(int)
    df['second'] = df['second'].astype(int)

    df['HDD']           = df['first']*df['Layer1HDD']           + df['second']*df['Layer2HDD']
    df['SSD']           = df['first']*df['Layer1SSD']           + df['second']*df['Layer2SSD']
    df['Hybrid']        = df['first']*df['Layer1Hybrid']        + df['second']*df['Layer2Hybrid']
    df['Flash_Storage'] = df['first']*df['Layer1Flash_Storage'] + df['second']*df['Layer2Flash_Storage']

    df.drop(columns=['Memory', 'first', 'second',
                     'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage',
                     'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage',
                     'Hybrid', 'Flash_Storage'], inplace=True)

    df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
    df = df[df['Gpu brand'] != 'ARM']
    df.drop(columns=['Gpu'], inplace=True)

    def cat_os(inp):
        if inp in ('Windows 10', 'Windows 7', 'Windows 10 S'):
            return 'Windows'
        elif inp in ('macOS', 'Mac OS X'):
            return 'Mac'
        else:
            return 'Others/No OS/Linux'

    df['os'] = df['OpSys'].apply(cat_os)
    df.drop(columns=['OpSys'], inplace=True)

    X = df.drop(columns=['Price'])
    y = np.log(df['Price'])

    # sparse_output replaces sparse in sklearn >= 1.2
    step1 = ColumnTransformer(transformers=[
        ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11])
    ], remainder='passthrough')

    step2 = RandomForestRegressor(n_estimators=100, random_state=3,
                                  max_samples=0.5, max_features=0.75,
                                  max_depth=15)

    pipe = Pipeline([('step1', step1), ('step2', step2)])
    pipe.fit(X, y)

    return pipe, df


# ── LOAD ──────────────────────────────────────────────────────────────────────
with st.spinner("Training model… (first load only, then cached)"):
    pipe, df = train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    company     = st.selectbox('Brand',           sorted(df['Company'].unique()))
    type_name   = st.selectbox('Type',            sorted(df['TypeName'].unique()))
    ram         = st.selectbox('RAM (GB)',         [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight      = st.number_input('Weight (kg)',  min_value=0.5, max_value=5.0,
                                   value=1.5, step=0.1)
    touchscreen = st.selectbox('Touchscreen',     ['No', 'Yes'])
    ips         = st.selectbox('IPS Display',     ['No', 'Yes'])

with col2:
    screen_size = st.number_input('Screen Size (inches)',
                                   min_value=10.0, max_value=18.0,
                                   value=15.6, step=0.1)
    resolution  = st.selectbox('Screen Resolution', [
        '1920x1080', '1366x768', '1600x900',
        '3840x2160', '3200x1800', '2560x1440',
        '2560x1600', '2304x1440'
    ])
    cpu  = st.selectbox('CPU Brand',         sorted(df['Cpu brand'].unique()))
    hdd  = st.selectbox('HDD (GB)',          [0, 128, 256, 512, 1000, 2000])
    ssd  = st.selectbox('SSD (GB)',          [0, 8, 128, 256, 512, 1000])
    gpu  = st.selectbox('GPU Brand',         sorted(df['Gpu brand'].unique()))
    os   = st.selectbox('Operating System',  sorted(df['os'].unique()))

# ── PREDICT ───────────────────────────────────────────────────────────────────
st.markdown("---")
if st.button('💰 Predict Price', use_container_width=True):
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val         = 1 if ips         == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi   = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query_df = pd.DataFrame([[company, type_name, ram, weight,
                               touchscreen_val, ips_val, ppi,
                               cpu, hdd, ssd, gpu, os]],
                             columns=['Company', 'TypeName', 'Ram', 'Weight',
                                      'Touchscreen', 'Ips', 'ppi',
                                      'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os'])

    query_df['Ram']         = query_df['Ram'].astype('int32')
    query_df['Weight']      = query_df['Weight'].astype('float32')
    query_df['Touchscreen'] = query_df['Touchscreen'].astype('int64')
    query_df['Ips']         = query_df['Ips'].astype('int64')
    query_df['ppi']         = query_df['ppi'].astype('float64')
    query_df['HDD']         = query_df['HDD'].astype('int64')
    query_df['SSD']         = query_df['SSD'].astype('int64')

    predicted_price = int(np.exp(pipe.predict(query_df)[0]))
    st.success(f"### 🏷️ Estimated Price:  ₹ {predicted_price:,}")