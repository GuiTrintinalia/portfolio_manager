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



# Displaying Dividends
st.subheader("Dividends")
st.dataframe(msft.dividends)

# Displaying Splits
st.subheader("Splits")
st.dataframe(msft.splits)

# Displaying Capital Gains
st.subheader("Capital Gains")
# Checking if the method exists for the selected stock
if hasattr(msft, 'capital_gains'):
    st.dataframe(msft.capital_gains)
else:
    st.write("Capital Gains data not available for this stock.")

# Displaying Share Count
st.subheader("Share Count")
st.dataframe(msft.get_shares_full(start="2022-01-01", end=None))

# Displaying Financials
st.subheader("Income Statement")
st.dataframe(msft.income_stmt)

st.subheader("Quarterly Income Statement")
st.dataframe(msft.quarterly_income_stmt)

st.subheader("Balance Sheet")
st.dataframe(msft.balance_sheet)

st.subheader("Quarterly Balance Sheet")
st.dataframe(msft.quarterly_balance_sheet)

st.subheader("Cash Flow Statement")
st.dataframe(msft.cashflow)

st.subheader("Quarterly Cash Flow Statement")
st.dataframe(msft.quarterly_cashflow)

# Displaying Holders
st.subheader("Major Holders")
st.dataframe(msft.major_holders)

st.subheader("Institutional Holders")
st.dataframe(msft.institutional_holders)

st.subheader("Mutual Fund Holders")
st.dataframe(msft.mutualfund_holders)

st.subheader("Insider Transactions")
st.dataframe(msft.insider_transactions)

st.subheader("Insider Purchases")
st.dataframe(msft.insider_purchases)

st.subheader("Insider Roster Holders")
st.dataframe(msft.insider_roster_holders)

# Displaying Recommendations
st.subheader("Recommendations")
st.dataframe(msft.recommendations)

st.subheader("Recommendations Summary")
st.dataframe(msft.recommendations_summary)

st.subheader("Upgrades/Downgrades")
st.dataframe(msft.upgrades_downgrades)

