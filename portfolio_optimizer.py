import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import plotly.express as px
import requests
import json
import time
import pickle
import datetime
from datetime import datetime, timedelta, date
from itertools import combinations
from multiprocessing import Pool
import time
import riskfolio as rp
import yfinance  as yf
import plotly.graph_objects as go

def candlestick_chart(dfs, selected_var):
    suffixes = ['Close_', 'Open_', 'Low_', 'High_']
    candles = []

    # Criar todos os sufixos
    for var in selected_var:
        for suffix in suffixes:
            column_name = f"{suffix}{var}"
            candles.append(column_name)

    # Filtrar o DataFrame com base nos column_names criados
    dfs = dfs[candles]
    traces = []

    for var in selected_var:
        trace = go.Candlestick(
            x=dfs.index,
            open=dfs[f"Open_{var}"],
            high=dfs[f"High_{var}"],
            low=dfs[f"Low_{var}"],
            close=dfs[f"Close_{var}"],
            name=var
        )

        traces.append(trace)

    layout = go.Layout(
        title="Gráfico de Candlestick",
        xaxis=dict(title="Data"),
        yaxis=dict(title="Preço"),
    )

    fig = go.Figure(data=traces, layout=layout)
    return fig

## Configuração da página e do título
st.set_page_config(page_title='Rebalanceador de carteira', layout = 'wide', initial_sidebar_state = 'auto')
st.title("Rebalanceador de carteira")

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(df=None, data=None, cleaned=False, messages=None)

session_state = get_session()

st.subheader('Crie sua carteira',divider='rainbow')

tipo_dados = st.sidebar.button('Tipo de dados', ['info','history','actions'])

# Get Tickers from yahoo finance
msft = yf.Ticker("MSFT")
hist = msft.history(period="1mo")
actions = msft.actions

st.dataframe(hist)
st.dataframe(actions)

