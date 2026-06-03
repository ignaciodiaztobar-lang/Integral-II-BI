import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="ConecTel - Predictor de Morosidad", layout="centered")

# Cargar archivos serializados
model = joblib.load('modelo_mora.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('features_list.pkl')

st.title("🛡️ Sistema de Alerta Temprana de Morosidad")
st.markdown("Ingrese los datos del cliente para evaluar su riesgo de mora severa (>90 días).")

# Crear el formulario de entrada
with st.form("formulario_cliente"):
    st.subheader("Datos del Cliente")
    col1, col2 = st.columns(2)
    
    with col1:
        valor_cliente = st.number_input("Valor Cliente (Estimado)", value=50000)
        edad = st.number_input("Edad del Cliente", 18, 100, 35)
        antiguedad = st.slider("Antigüedad (meses)", 0, 120, 12)
        
    with col2:
        dias_mora_hist = st.number_input("Días de mora históricos", 0, 365, 0)
        reclamos = st.number_input("Reclamos en último año", 0, 20, 0)
        num_servicios = st.selectbox("Número de servicios", [1, 2, 3])

    # El resto de variables que pide tu modelo deben estar presentes
    # Para las que no pedimos por UI, pondremos valores por defecto (0 o promedio)
    
    enviar = st.form_submit_button("Analizar Riesgo")

if enviar:
    # 1. Crear el diccionario con los datos del usuario
    # IMPORTANTE: Los nombres deben ser EXACTOS a como aparecen en X_mi.columns
    # Si alguno es un 'dummy' (ej: metodo_pago_Efectivo), debes ponerlo también
    datos_usuario = {
        'valor_cliente': valor_cliente,
        'edad': edad,
        'antiguedad_meses': antiguedad,
        'dias_mora_hist': dias_mora_hist,
        'reclamos_12m': reclamos,
        'num_servicios': num_servicios,
        # Agrega aquí el resto de las variables que falten de tu top 10
    }

    # 2. Crear DataFrame y asegurar que el ORDEN sea el mismo de Colab
    input_df = pd.DataFrame([datos_usuario])
    
    # Rellenamos con 0 cualquier variable de features_list que no hayamos pedido en el UI
    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reordenamos las columnas exactamente como las espera el scaler/modelo
    input_df = input_df[features]

    try:
        # 3. Escalar usando el nuevo scaler sincronizado
        input_scaled = scaler.transform(input_df)
        
        # 4. Predicción
        probabilidad = model.predict_proba(input_scaled)[0][1]
        
        # Mostrar resultados
        st.subheader(f"Probabilidad de Mora: {probabilidad:.1%}")
        if probabilidad < 0.3:
            st.success("Riesgo Bajo")
        elif probabilidad < 0.7:
            st.warning("Riesgo Medio")
        else:
            st.error("Riesgo Alto")
            
    except Exception as e:
        st.error(f"Error técnico: {e}")
