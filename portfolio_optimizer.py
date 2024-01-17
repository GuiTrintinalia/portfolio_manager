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
import io
from io import BytesIO
import pyarrow.parquet as pq


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

def donwload_data(tickers, period): 
    dfs = []
    
    for ticker in tickers:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=period)
        hist.columns = [f"{ticker}_{col}" for col in hist.columns]  # Adicionar prefixo ao ticker
        dfs.append(hist)

    df = pd.concat(dfs, axis=1)  # Concatenar pelo índice de datas
    return df

def load_data_from_github(url):
    response = requests.get(url)
    content = BytesIO(response.content)
    data = pd.read_pickle(content)
    return data


## Configuração da página e do título
st.set_page_config(page_title='Wallet Balancer', layout = 'wide', initial_sidebar_state = 'auto')
st.title("Wallet Balancer")

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(df=None, data=pd.DataFrame())
session_state = get_session()

st.subheader('Crie sua carteira',divider='rainbow')
type_tickers = st.text_input('Type stock separed by comma ( AAPL, MSFT, ...):')

url = 'https://github.com/GuiTrintinalia/portfolio_manager/raw/main/tickers.pkl'

if st.button('Load Tickers'):
    yahoo_list = load_data_from_github(url)
    yahoo_list = list(yahoo_list.values())
    tickers_list = st.multiselect('Available Tickers:', yahoo_list)
else:
    st.info("Pressione o botão para carregar os tickers.")


if tickers_list:
    tickers = tickers_list
else:
    tickers = [ticker.strip() for ticker in type_tickers.split(',')]

selected_timeframe = st.selectbox('Select Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])

if st.button("Download data"):
    session_state.data = donwload_data(tickers, selected_timeframe)
    if session_state.data is not None:
        st.dataframe(session_state.data)



