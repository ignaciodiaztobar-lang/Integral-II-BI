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
    # 1. Crear un diccionario con TODAS las variables que el MODELO espera (las 10 de X_mi)
    # Deben tener exactamente el mismo nombre que en tu Colab
    datos_para_modelo = pd.DataFrame(columns=features) # Usamos la lista de features que guardamos
    
    # 2. Llenamos una fila con los datos de la interfaz
    # Asegúrate de que los nombres coincidan con los de tu gráfico de importancia
    nueva_fila = {f: 0 for f in features} # Llenamos todo con 0 por defecto
    
    # Mapeamos lo que el usuario ingresó (ajusta los nombres a tus columnas de X_mi)
    nueva_fila['valor_cliente'] = valor_cliente
    nueva_fila['edad'] = edad
    nueva_fila['antiguedad_meses'] = antiguedad
    nueva_fila['dias_mora_hist'] = dias_mora_hist
    nueva_fila['reclamos_12m'] = reclamos
    nueva_fila['num_servicios'] = num_servicios
    
    # Si tienes variables categóricas en tu top 10 (como metodo_pago_Efectivo), 
    # aquí deberías poner lógica de 1 o 0.
    
    # Convertimos a DataFrame
    input_df = pd.DataFrame([nueva_fila])

    # 3. EL TRUCO PARA EL SCALER:
    # El Scaler falló porque espera TODAS las columnas numéricas que vio en Colab.
    # Vamos a hacer una pequeña trampa: solo escalaremos si es estrictamente necesario, 
    # o nos aseguraremos de pasarle solo las columnas que él conoce.
    
    try:
        # Intento de transformar (esto fallaba)
        # Si tu scaler solo se entrenó con las numéricas, el input_df debe tener SOLO esas columnas para el transform
        input_scaled = scaler.transform(input_df[features]) 
        
        # 4. Predicción
        probabilidad = model.predict_proba(input_scaled)[0][1]
        
        # Mostrar resultados (Esto sigue igual...)
        st.subheader(f"Probabilidad de Mora: {probabilidad:.1%}")
        # ... resto del código de st.success/error
        
    except Exception as e:
        st.error(f"Error técnico de alineación: {e}")
        st.info("Asegúrate de que las columnas en app.py coincidan con las de features_list.pkl")
