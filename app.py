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
    # 1. Crear un DataFrame con TODAS las variables que espera el modelo en orden
    # Nota: Aquí debes asegurar que el diccionario tenga las mismas llaves que 'features'
    datos = {f: 0 for f in features} # Inicializamos todo en 0
    datos['valor_cliente'] = valor_cliente
    datos['edad'] = edad
    datos['antiguedad_meses'] = antiguedad
    datos['dias_mora_hist'] = dias_mora_hist
    datos['reclamos_12m'] = reclamos
    datos['num_servicios'] = num_servicios
    
    input_df = pd.DataFrame([datos])
    
    # 2. Escalar los datos (usando el scaler de Colab)
    # Debemos seleccionar solo las columnas que el scaler conoce
    # (Asegúrate de que 'columnas_a_escalar' sea igual a la de tu Colab)
    cols_to_scale = ['edad', 'antiguedad_meses', 'num_servicios', 'dias_mora_hist', 'reclamos_12m'] # Ejemplo
    # Nota: El scaler espera todas las columnas originales. Una forma simple:
    input_scaled = scaler.transform(input_df) # Ajustar según tu scaler específico
    
    # 3. Predicción
    probabilidad = model.predict_proba(input_scaled)[0][1]
    
    # 4. Mostrar resultados
    st.divider()
    st.subheader(f"Probabilidad de Mora: {probabilidad:.1%}")
    
    if probabilidad < 0.3:
        st.success("RIESGO BAJO: Cliente con buen comportamiento de pago.")
    elif probabilidad < 0.7:
        st.warning("RIESGO MEDIO: Se recomienda monitoreo proactivo.")
    else:
        st.error("RIESGO ALTO: Acción de cobranza inmediata requerida.")