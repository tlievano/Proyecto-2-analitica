import dash
from dash import dcc  # dash core components
from dash import html # dash html components
from dash.dependencies import Input, Output
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from dotenv import load_dotenv # pip install python-dotenv
import os
import psycopg2
from tensorflow.keras.models import load_model
import pandas.io.sql as sqlio
from dash import Dash, dcc, html
from dash.dependencies import Input, Output  # Asegúrate de que Input y Output provengan de esta línea




external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


# path to env file
#env_path="/Users/sofiabuitrago/Desktop/app.env"
# load env 
#load_dotenv(dotenv_path=env_path)
# extract env variables
#BDUSER=os.getenv('BDUSER')
#PASSWORD=os.getenv('PASSWORD')
#HOST=os.getenv('HOST')
#PORT=os.getenv('PORT')
#DBNAME=os.getenv('DBNAME')

# Cargar el modelo .h5
#model = load_model('/Users/sofiabuitrago/Desktop/modelo.keras')

#cursor = engine.cursor()
query = """
SELECT * 
FROM proy2;"""
df = pd.read_csv("/Users/sofiabuitrago/Desktop/bank-full-modelo.csv", delimiter=",")


# Función para crear gráficos de las variables categóricas
def create_categorical_graphs():
    graphs = []

    # Lista de variables categóricas y sus títulos
    categorical_vars = {
        'marital': 'Distribución del Estado Civil',
        'education': 'Distribución del Nivel de Educación',
        'job': 'Distribución de Profesiones',
        'month': 'Distribución de Meses de Contacto'
    }

    # Colores personalizados
    colors = {
        'marital': ['#1f77b4', '#ff7f0e', '#2ca02c'], 
        'education': ['#d62728', '#9467bd', '#8c564b', '#e377c2'],  
        'job': ['#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'], 
        'month': ['#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#f7b1c8', '#e1f9b8']  
    }

    # Crear gráficos para cada variable categórica
    for var, title in categorical_vars.items():
        fig = px.histogram(df, x=var, title=title, color=var,
                           color_discrete_sequence=colors[var],
                           labels={var: title})
        fig.update_layout(xaxis_title=title, yaxis_title='Conteo')
        graphs.append(html.Div([
            dcc.Graph(figure=fig),
        ], style={'margin-bottom': '30px'}))
    
    return graphs

# Función para calcular la predicción
def calculate_prediction(marital_status):
    if marital_status == "Casado":
        return 0.5720
    elif marital_status == "Soltero":
        return 0.3245
    elif marital_status == "Divorciado":
        return 0.1828
    else:
        return None  

# Layout de la aplicación Dash
app.layout = html.Div([
    html.H1("Predicción de Depósito Bancario", style={'font-family': 'Calibri Light'}),

    html.Div([
        # Estado Civil
        html.Div([
            html.Label('Estado Civil'),
            dcc.Dropdown(
                id='marital-dropdown',
                options=[
                    {'label': 'Divorciado', 'value': 'Divorciado'},
                    {'label': 'Casado', 'value': 'Casado'},
                    {'label': 'Soltero', 'value': 'Soltero'}
                ],
                value='Divorciado'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Nivel de Educación
        html.Div([
            html.Label('Nivel de Educación'),
            dcc.Dropdown(
                id='education-dropdown',
                options=[
                    {'label': 'Primaria', 'value': 'Primaria'},
                    {'label': 'Secundaria', 'value': 'Secundaria'},
                    {'label': 'Terciaria', 'value': 'Terciaria'},
                    {'label': 'Desconocida', 'value': 'Desconocida'}
                ],
                value='Primaria'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Profesión
        html.Div([    
            html.Label('Profesión'),
            dcc.Dropdown(
                id='job-dropdown',
                options=[
                    {'label': 'Admin', 'value': 'Admin'},
                    {'label': 'Blue-collar', 'value': 'Blue-collar'},
                    {'label': 'Entrepreneur', 'value': 'Entrepreneur'},
                    {'label': 'Housemaid', 'value': 'Housemaid'},
                    {'label': 'Management', 'value': 'Management'},
                    {'label': 'Retired', 'value': 'Retired'},
                    {'label': 'Self-employed', 'value': 'Self-employed'},
                    {'label': 'Services', 'value': 'Services'},
                    {'label': 'Student', 'value': 'Student'},
                    {'label': 'Technician', 'value': 'Technician'},
                    {'label': 'Unemployed', 'value': 'Unemployed'},
                    {'label': 'Unknown', 'value': 'Unknown'}
                ],
                value='Admin'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Edad
        html.Div([
            html.Label('Edad'),
            dcc.Input(id='age-input', type='number', min=18, max=95, step=1, value=30),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Mes de contacto
        html.Div([    
            html.Label('Mes de contacto'),
            dcc.Dropdown(
                id='month-dropdown',
                options=[
                    {'label': 'Enero', 'value': 'Enero'},
                    {'label': 'Febrero', 'value': 'Febrero'},
                    {'label': 'Marzo', 'value': 'Marzo'},
                    {'label': 'Abril', 'value': 'Abril'},
                    {'label': 'Mayo', 'value': 'Mayo'},
                    {'label': 'Junio', 'value': 'Junio'},
                    {'label': 'Julio', 'value': 'Julio'},
                    {'label': 'Agosto', 'value': 'Agosto'},
                    {'label': 'Septiembre', 'value': 'Septiembre'},
                    {'label': 'Octubre', 'value': 'Octubre'},
                    {'label': 'Noviembre', 'value': 'Noviembre'},
                    {'label': 'Diciembre', 'value': 'Diciembre'}
                ],
                value='Enero'
            ),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Variables continuas adicionales
        html.Div([
            html.Label('Balance'),
            dcc.Input(id='balance-input', type='number', value=500),
        ], style={'width': '25%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Duration'),
            dcc.Input(id='duration-input', type='number', value=100),
        ], style={'width': '25%', 'display': 'inline-block'}),

        html.Div([
            html.Label('Campaign'),
            dcc.Input(id='campaign-input', type='number', value=1),
        ], style={'width': '25%', 'display': 'inline-block'}),

        # Botón de predicción
        html.Button('Predecir', id='predict-button', n_clicks=0),

        # Div para mostrar el resultado de la predicción
        html.Div(id='prediction-output', style={'margin-top': '20px', 'font-size': '20px'}),

        # Sección para mostrar gráficos
        html.Div(id='graphs-container', style={'margin-top': '40px'},
                 children=create_categorical_graphs()),  # Gráficas en el layout
    ])  
]) 


# Callback para realizar la predicción
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('marital-dropdown', 'value')
)
def update_prediction(n_clicks, marital_status):
    if n_clicks > 0:
        prediction_value = calculate_prediction(marital_status) 

        if prediction_value > 0.5:
            mensaje = "el cliente contratará un depósito."
        else:
            mensaje = "el cliente no contratará un depósito."
        
        return f'Con una probabilidad de: {round(prediction_value*100)}%, {mensaje}'
    return "Haz clic en 'Predecir' para ver el resultado."



# Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)
