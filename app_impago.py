import pandas as pd
import streamlit as st
import numpy as np
from pycaret.classification import load_model, predict_model

st.set_page_config(page_title="Modelo de Clasificacion de Impago de Impuesto Predial")

@st.cache(allow_output_mutation=True)
def get_model():
    return load_model("C:/Users/gchav/OneDrive/Escritorio/Deployment_Impago/modelo_PyCaret_Version_VC")

def predict(model, df):
    predictions = predict_model(model, data = df)
    return predictions['Label'][0]

model = get_model()

st.title("Modelo de Clasificiación de Impago de Impuesto Predial")
st.markdown("Elija los valores para cada atributo del modelo de Impago de Impuesto Predial\
            para clasificar el predio en Vigente(1) o No Vigente(0)")

form = st.form("predios")
VALOR_CATASTRAL = form.number_input('Valor Catastral', min_value = 59.0 , max_value = 900000000.00,value=500000.00 , format = '%.2f', step = 101.00)
regimen_list = ['BALDIO SIN BARDAR','PARTICULARES','FEDERALES','MUNICIPALES','ESCUELAS','ESTATALES','CENTRO HISTORICO COMERCIAL','CLAVE CATASTRAL BLOQUEADA']
REGIMEN = form.selectbox('Regimen', regimen_list)
giro_list = ['BALDIO','INDUSTRIAL','HABITACIONAL','DEPENDENCIAS FEDERALES','COMERCIAL','C. RECREATIVOS','ESCUELA','BALDIO BARDADO','SERVICIOS','MUNICIPAL','AREAS VERDES','RURAL','CENTRO DE SALUD','USO AGRICOLA','ABASTO','CENTROS RELIGIOSOS','FEDERAL','PRIVADO','SIN USO','ALBERGUE','ROTACION','ESTATAL','ERIAZO','DEPARTAMENTOS','COMUNICACIONES','BOMBA DE AGUA','HUERTO','ESTACIONAMIENTO']
GIRO = form.selectbox('Giro', giro_list)
Nse_list = ['Muy Alto', 'Alto', 'Medio', 'Bajo', 'Muy Bajo', 'SD']
NSE = form.selectbox('NSE', Nse_list)
Cuadrante_list = ['NE', 'NO', 'SE', 'SO']
Cuadrante = form.selectbox('Cuadrante', Cuadrante_list)
Distance_Km = form.slider('Distance Km', min_value = 0.0 , max_value = 15.0 , value = 2.0, format = '%.2f' )

predict_button = form.form_submit_button('Predict')

input_dict = {'VALOR_CATASTRAL': VALOR_CATASTRAL, 'REGIMEN': REGIMEN, 'GIRO': GIRO, 'NSE': NSE, 'Cuadrante': Cuadrante, 'Distance_Km': Distance_Km}

input_df = pd.DataFrame([input_dict])

if predict_button: 
    out = predict(model, input_df)
    st.success(f'La prediccion del predio es {out}.')

