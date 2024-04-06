# basic pyhton librarioes
import streamlit as st
import pandas as pd
import numpy as np
from datetime import  date

# libraries to retrieve and download data
import requests
from io import BytesIO
import yfinance  as yf
import base64

# ploting libraries
import plotly.express as px
import plotly.graph_objects as go

# for combinations
import itertools



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

def download_data(data, period='1y'):
    dfs = []
    if isinstance(data, dict):
        for name, ticker in data.items():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{name}_{col}" for col in hist.columns]  # Add prefix to the name
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist)
    elif isinstance(data, list):
        for ticker in data:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{ticker}_{col}" for col in hist.columns]  # Add prefix to the name
            hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
            dfs.append(hist)

    combined_df = pd.concat(dfs, axis=1, join='outer')  # Use join='outer' to handle different time indices
    return combined_df

def upload_file(file):
    df = pd.read_excel(file) if file.name.endswith('xlsx') else pd.read_csv(file)
    df.set_index('Date', inplace = True, drop = True)
    df.index = pd.to_datetime(df.index)
    return df

def download_dfs(session_state, download_option, mapping):
    attribute_name = mapping.get(download_option)
    if attribute_name is not None:
        data_attribute = getattr(session_state, attribute_name)
        if data_attribute is not None:
            df_to_download = pd.DataFrame(data_attribute)
            csv = df_to_download.to_csv(index=True)  # incluir o índice no arquivo CSV
            b64 = base64.b64encode(csv.encode()).decode()  # codificação B64 para o link de download
            href = f'<a href="data:file/csv;base64,{b64}" download="{download_option}.csv">Download {download_option} Data</a>'
            return href
        else:
            st.warning(f"No data available for {download_option}.")
            return None
    else:
        st.error(f"Invalid download option: {download_option}")
        return None
    
def load_data_from_github(url):
    response = requests.get(url)
    content = BytesIO(response.content)
    data = pd.read_pickle(content)
    return data

def date_resample(df, period='M', aggregation='sum'):
    result_series = df.resample(period).agg(aggregation)
    return result_series

def fill_moving_avg(df, window_size, method='gap'):
    if method == 'gap':
        date_index = df.index
        df.reset_index(drop=True, inplace=True)
        for col in df.select_dtypes(include=[np.number]).columns:
            nan_indices = df[df[col].isna()].index
            for index in nan_indices:
                start = max(0, index - window_size)
                end = index + 1
                window_data = df[col].iloc[start:end]
                mean_value = round(window_data.mean(), 4)
                df.at[index, col] = mean_value
        df.index = date_index
    else:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            df[col] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[col] = df[col].fillna(method='bfill')
    return df

def drop_nan_rows(df):
    df_cleaned = df.dropna()
    return df_cleaned
	
def get_latest_values(df, tickers):
    latest_values = {}
    for ticker in tickers:
        latest_index = df[f"{ticker}_Close"].idxmax()
        latest_value = df.at[latest_index, f"{ticker}_Close"]
        latest_values[ticker] = latest_value
    
    return latest_values

def compute_investments(df, tickers, total_shares, available_cash):
    latest_values = get_latest_values(df, tickers)
    investments = pd.DataFrame(columns=['Ticker', 'Price', 'Papers', 'Share %', 'Invested', 'Cash'])

    cash_to_invest = available_cash  # Initialize cash_to_invest

    for i, ticker in enumerate(tickers):
        share = total_shares[i]
        price = latest_values[ticker]
        invested = (share * available_cash)
        papers = invested / price
        cash_to_invest -= invested  # Subtract the invested amount for each iteration

        row_data = {'Ticker': ticker, 'Price': price, 'Invested': invested, 
                    'Share %': share*100, 'Papers': papers, 'Cash': round(cash_to_invest, 2)}
        investments = pd.concat([investments, pd.DataFrame([row_data])], ignore_index=True)

    return investments[['Ticker', 'Price', 'Papers', 'Share %', 'Invested', 'Cash']]

def logreturns(df):
    df.columns = df.columns.str.split('_').str[0]
    log_returns  = np.log(df)
    log_returns = df.iloc[:, 0:].pct_change()
    fig = px.line(log_returns, x=log_returns.index, y=log_returns.columns[0:],
                  labels={'value': 'log'},
                  title='log returns')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig)

    return log_returns

def return_over_time(df):
    return_df = pd.DataFrame()
    df.columns = df.columns.str.split('_').str[0]
    for col in df.columns:
        return_df[col] = df[col] / df[col].iloc[0] -1
        
    fig = px.line(return_df, x=return_df.index, y=return_df.columns[0:],
                  labels={'value': 'Returns to Date'},
                  title='returns')
    fig.update_layout(legend_title_text='Assets')
    st.plotly_chart(fig)
    
    
def efficient_frontier(df, trading_days,total_shares, risk_free_rate, simulations= 1000, resampler='A'):
    return_over_time(df)
    logreturns(df)
    
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    portfolio_var = cov_matrix.mul(total_shares,axis=0).mul(total_shares,axis=1).sum().sum()
    standard_deviation = np.sqrt(portfolio_var) 
    annualized_risk  = standard_deviation *np.sqrt(trading_days)
    simple_annualized_risk = np.exp(annualized_risk) - 1
    simple_risk = np.exp(annualized_risk) - 1
    st.markdown(f'Annualized portfolio risk:  **{simple_risk:.4f}**')
    annualized_returns = df.resample(resampler).last()
    st.dataframe(annualized_returns)
    annualized_returns = df.pct_change().apply(lambda x: np.log(1 + x)).mean() * trading_days
    annualized_returns = annualized_returns.rename('Log Returns')
    simple_returns = np.exp(annualized_returns) - 1
    simple_returns = simple_returns.rename('Simple Returns')
    weights_series = pd.Series(total_shares, index=df.columns, name='Weights')
    returns_df = pd.concat([annualized_returns, simple_returns, weights_series], axis=1)
    returns_df['Weighted Returns'] = returns_df['Simple Returns'] * returns_df['Weights']
    returns_df['Cumulative Portfolio Returns'] = returns_df['Weighted Returns'].cumsum()

    st.markdown(f'**Annualized portfolio return:**')
    st.dataframe(returns_df)
    assets_return = simple_returns *total_shares
    portfolio_return = assets_return.sum()
    st.markdown(f'Annualized portfolio return: **{portfolio_return:.4f}**')
    sharpe_ratio = (portfolio_return - risk_free_rate)/ simple_annualized_risk
    st.markdown(f'Initial allocation Sharpe Ratio  **{sharpe_ratio:.4f}**')

    portfolio_returns = [] 
    portfolio_variance = [] 
    portfolio_weights = [] 
    
    num_assets = len(df.columns)    

    for _ in range(simulations):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        portfolio_weights.append(weights)
        returns = np.dot(weights, simple_returns) # Returns are the product of individual expected returns of asset and its 
        portfolio_returns.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(trading_days) # Annual standard deviation = volatility
        portfolio_variance.append(ann_sd)

        data = {'Returns':portfolio_returns, 'Volatility':portfolio_variance}

    for counter, symbol in enumerate(df.columns.tolist()):
        data[symbol+' weight'] = [w[counter] for w in portfolio_weights]
    
    simulated_portfolios  = pd.DataFrame(data)
    simulated_portfolios = simulated_portfolios.round(4)

    return simulated_portfolios

def plot_efficient_frontier(simulated_portfolios, risk_free_rate, expected_sharpe, expected_return, risk_taken):
    simulated_portfolios = simulated_portfolios.sort_values(by='Volatility')
    simulated_portfolios['Sharpe_ratio'] = (simulated_portfolios['Returns'] - risk_free_rate) / simulated_portfolios['Volatility']
    
    simulated_portfolios['Weights'] = simulated_portfolios.iloc[:, 2:-1].apply(
        lambda row: ', '.join([f"{asset}: {weight:.4f}" for asset, weight in zip(simulated_portfolios.columns[2:-1], row)]),
        axis=1
    )

    frontier = px.scatter(simulated_portfolios, x='Volatility', y='Returns', width=800, height=600, 
                          title="Markowitz's Efficient Frontier", labels={'Volatility': 'Volatility', 'Returns': 'Return'},
                          hover_name='Weights')
    
    max_sharpe_ratio_portfolio = simulated_portfolios.loc[simulated_portfolios['Sharpe_ratio'].idxmax()]
    frontier.add_trace(go.Scatter(x=[max_sharpe_ratio_portfolio['Volatility']], 
                                  y=[max_sharpe_ratio_portfolio['Returns']],
                                  mode='markers',
                                  marker=dict(color='green', size=10),
                                  name='Max Sharpe Ratio',
                                  text=max_sharpe_ratio_portfolio['Weights']))

    low_risk_portfolios = simulated_portfolios[
        (simulated_portfolios['Returns'] >= expected_return) & 
        (simulated_portfolios['Volatility'] <= risk_taken)
    ]
    
    frontier.add_trace(go.Scatter(x=low_risk_portfolios['Volatility'], 
                                  y=low_risk_portfolios['Returns'],
                                  mode='markers',
                                  marker=dict(color='purple', size=5),
                                  name='Expected Return & Risk Taken',
                                  text=low_risk_portfolios['Weights']))
    
    expected_portfolio = simulated_portfolios[
        (simulated_portfolios['Sharpe_ratio'] >= expected_sharpe - 0.001) & 
        (simulated_portfolios['Sharpe_ratio'] <= expected_sharpe + 0.001)
    ]
    
    if not expected_portfolio.empty:
        frontier.add_trace(go.Scatter(x=[expected_portfolio['Volatility'].values[0]], 
                                      y=[expected_portfolio['Returns'].values[0]],
                                      mode='markers',
                                      marker=dict(color='orange', size=10),
                                      name='Expected Sharpe Ratio',
                                      text=expected_portfolio['Weights']))
    
    frontier.add_trace(go.Scatter(x=[simulated_portfolios.iloc[0]['Volatility']], 
                                  y=[simulated_portfolios.iloc[0]['Returns']],
                                  mode='markers',
                                  marker=dict(color='red', size=10),
                                  name='Min Volatility', 
                                  text=simulated_portfolios.iloc[0]['Weights']))
    
    frontier.update_layout(legend=dict(
                           orientation='h',
                           yanchor='top',
                           y=1.1,
                           xanchor='center',
                           x=0.5))
    
    max_sharpe_ratio_value = simulated_portfolios['Sharpe_ratio'].max()
    st.markdown(f'Max Sharpe Ratio: **{max_sharpe_ratio_value:.2f}**')
    st.plotly_chart(frontier)
    
    
def backtest_frontier(df_list, risk_free_rate, trading_days, simulations=1000):
    result_dfs = []
    for df in df_list:
        cov_matrix = df.drop(columns=['ID', 'date']).pct_change().apply(lambda x: np.log(1 + x)).cov()
        portfolio_returns = [] 
        portfolio_variance = [] 
        portfolio_weights = [] 
        portfolio_sharpe_ratio =[]
        num_assets = len(df.columns)-2  # Subtracting 'ID' and 'date' columns

        annualized_returns = df.drop(columns=['ID', 'date']).pct_change().apply(lambda x: np.log(1 + x)).mean() * trading_days
        for _ in range(simulations):
            weights = np.random.random(num_assets)
            weights = weights/np.sum(weights)
            returns = np.dot(weights, annualized_returns)
            portfolio_weights.append(weights)
            portfolio_returns.append(returns)
            var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  
            sd = np.sqrt(var) 
            ann_sd = sd * np.sqrt(trading_days)  
            portfolio_variance.append(ann_sd)
            if returns > 0:
                sharpe_ratio = (returns - np.log(1 + risk_free_rate)) / ann_sd
            else:
                sharpe_ratio = (returns + np.log(1 + risk_free_rate)) / ann_sd

            portfolio_sharpe_ratio.append(sharpe_ratio)

        data = {
            'Log Returns': portfolio_returns,
            'Volatility': portfolio_variance,
            'Sharpe_ratio': portfolio_sharpe_ratio,
            'ID': df['ID'].iloc[0],
            'Date': df['date'].iloc[0]  # Insert starting day for each ID
        }

        for counter, symbol in enumerate(df.columns):
                if symbol == 'ID' or symbol == 'date':
                    continue
                ticker_name = symbol.split('_')[0]  # Remove underscores from ticker name
                data[ticker_name + '_Weight'] = [w[counter] for w in portfolio_weights]
            
        simulated_portfolios = pd.DataFrame(data)
        simulated_portfolios = simulated_portfolios.round(4)
        result_dfs.append(simulated_portfolios)
        
    final_df = pd.concat(result_dfs, ignore_index=True)

    return final_df
def get_max_sharpe_per_id(final_df, max_values, min_values):
    weight_columns = [column for column in final_df.columns if column.endswith('_Weight')]
    num_columns = len(weight_columns)

    filtered_indices = set(range(len(final_df)))

    for i in range(num_columns):
        weight_column = weight_columns[i]
        min_value = min_values[i]
        max_value = max_values[i]

        filtered_indices_temp = set(index for index, row in final_df.iterrows() if min_value <= row[weight_column] <= max_value)
        filtered_indices.intersection_update(filtered_indices_temp)

    filtered_df = final_df.iloc[list(filtered_indices)]
    max_sharpe_rows = filtered_df.loc[filtered_df.groupby('ID')['Sharpe_ratio'].idxmax()]
    
    return max_sharpe_rows

def optimizeBySharpe(df):
    pd.options.display.float_format = '{:.3f}'.format
    pricesT2 = df['pricesT2'].values
    qtT1 = df['qtT1'].values
    weightsT2 = df['weightsT2'].values
    pricesT1 = df['pricesT1'].values
    investedT1 = pricesT1 * qtT1
    totalInvestedT1 = np.sum(investedT1)
    fund = qtT1 * pricesT2
    funds = np.sum(fund)
    maxQtT2 = funds * weightsT2 / pricesT2
    maxInvestedT2 = maxQtT2 * pricesT2
    optQtT2 = np.sum(maxInvestedT2) * weightsT2 /  pricesT2
    optWeightsT2 = optQtT2 * pricesT2 / np.sum(maxInvestedT2)
    otimizationROI = (funds /totalInvestedT1)
    portfolioROI = funds / invested_cash
    df['optWeightsT2'] = optWeightsT2
    df['optQtT2'] = optQtT2
    df['qtBuyOrSell'] = abs(optQtT2 - qtT1)
    df['investedT1'] = investedT1
    df['investedT2'] = fund
    
    df = df[['pricesT1', 'pricesT2', 'investedT1', 'investedT2',
             'qtT1', 'optQtT2', 'qtBuyOrSell', 'weightsT1', 
             'optWeightsT2', 'weightsT2']]
    
    st.markdown(f'**Total Invested T1:** {totalInvestedT1:.3f}')
    st.markdown(f'**Available Cash T2:** {funds:.3f}')
    st.markdown(f'**Otimization ROI:** {otimizationROI:.3f}')
    st.markdown(f'**Portfolio ROI:** {portfolioROI:.3f}')
    st.dataframe(df)
    
    return df

def returns_till_date(df, invested_cash, price_columns, weight_columns):
    first_row = df.iloc[0]
    initial_quantities_per_asset = []
    for price_col, weight_col in zip(price_columns, weight_columns):
        initial_quantity = invested_cash * first_row[weight_col] / first_row[price_col]
        initial_quantities_per_asset.append(initial_quantity)
        
    initial_quantities = np.array([initial_quantities_per_asset])
    initial_values = initial_quantities * df[price_columns].values
    invested_till_date = np.sum(initial_values, axis=1)
    return_till_date = invested_till_date / invested_cash
    ROItilldate = pd.DataFrame({'invested_till_date': invested_till_date, 'return_till_date': return_till_date}, index=df.index)
    st.dataframe(ROItilldate)
    return ROItilldate 

st.set_page_config(page_title='Portfolio Balancer', layout = 'wide', initial_sidebar_state = 'auto')
st.title("Portfolio Balancer")

# Importar SessionState
class SessionState:
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

@st.cache(allow_output_mutation=True)
def get_session():
    return SessionState(df=pd.DataFrame(), data=pd.DataFrame(), portfolio=pd.DataFrame(), backtest =pd.DataFrame(), optimized_data =pd.DataFrame())
session_state = get_session()

st.subheader('Pick your assets',divider='rainbow')

with st.expander("See explanation"):
    st.markdown("""
        1. You can combine assets from currencies, crypto, commodities, Nasdaq, S&P500, B3, and Indexes.
        
        2. Once you have selected one or more dictionaries, you will be able to choose tickers to download.
        
        3. You must press the download button to retrieve the data.
        
        4. It downloads data from Yahoo Finance, yet there may be tickers which return None or fragmented data.
        
        5. Use the resample and rolling average functions to prepare your data for optimization.
        
        6. If you prefer, download data with the first and last index for all tickers. However, this option may reduce the length of your resulting dataframe. It is optional to remove NaNs without any further data filling techniques.
        
        7. Additionally, you have the ability to:
            - Download your data as a CSV or Excel file for further analysis.
            - Upload your own portfolio data for educational purposes, allowing you to experiment with different optimization strategies.
    """)

file  = st.file_uploader("Upload Excel/CSV file", type=["xlsx", "csv"])
if file:
    session_state.data = upload_file(file)


currencies_dict  =  {'USD/JPY': 'USDJPY=X', 'USD/BRL': 'BRL=X', 'USD/ARS': 'ARS=X', 'USD/PYG': 'PYG=X', 'USD/UYU': 'UYU=X',
                     'USD/CNY': 'CNY=X', 'USD/KRW': 'KRW=X', 'USD/MXN': 'MXN=X', 'USD/IDR': 'IDR=X', 'USD/EUR': 'EUR=X',
                     'USD/CAD': 'CAD=X', 'USD/GBP': 'GBP=X', 'USD/CHF': 'CHF=X', 'USD/AUD': 'AUD=X', 'USD/NZD': 'NZD=X',
                     'USD/HKD': 'HKD=X', 'USD/SGD': 'SGD=X', 'USD/INR': 'INR=X', 'USD/RUB': 'RUB=X', 'USD/ZAR': 'ZAR=X',
                     'USD/SEK': 'SEK=X', 'USD/NOK': 'NOK=X', 'USD/TRY': 'TRY=X', 'USD/AED': 'AED=X', 'USD/SAR': 'SAR=X',
                     'USD/THB': 'THB=X', 'USD/DKK': 'DKK=X', 'USD/MYR': 'MYR=X', 'USD/PLN': 'PLN=X', 'USD/EGP': 'EGP=X',
                     'USD/CZK': 'CZK=X', 'USD/ILS': 'ILS=X', 'USD/HUF': 'HUF=X', 'USD/PHP': 'PHP=X', 'USD/CLP': 'CLP=X',
                     'USD/COP': 'COP=X', 'USD/PEN': 'PEN=X', 'USD/KWD': 'KWD=X', 'USD/QAR': 'USD/QAR'
                    }
crypto_dict = {'BITCOIN USD': 'BTC-USD', 'ETHEREUM USD': 'ETH-USD', 'TETHER USDT USD': 'USDT-USD',
               'BNB USD': 'BNB-USD', 'SOLANA USD': 'SOL-USD', 'XRP USD': 'XRP-USD', 'USD COIN USD': 'USDC-USD',
               'LIDO STAKED ETH USD': 'STETH-USD', 'CARDANO USD': 'ADA-USD', 'AVALANCHE USD': 'AVAX-USD',
               'DOGECOIN USD': 'DOGE-USD', 'WRAPPED TRON USD': 'WTRX-USD', 'TRON USD': 'TRX-USD',
               'POLKADOT USD': 'DOT-USD', 'CHAINLINK USD': 'LINK-USD', 'TONCOIN USD': 'TON11419-USD',
               'POLYGON USD': 'MATIC-USD', 'WRAPPED BITCOIN USD': 'WBTC-USD', 'SHIBA INU USD': 'SHIB-USD',
               'INTERNET COMPUTER USD': 'ICP-USD', 'DAI USD': 'DAI-USD', 'LITECOIN USD': 'LTC-USD',
               'BITCOIN CASH USD': 'BCH-USD', 'UNISWAP USD': 'UNI7083-USD', 'COSMOS USD': 'ATOM-USD',
               'UNUS SED LEO USD': 'LEO-USD', 'ETHEREUM CLASSIC USD': 'ETC-USD', 'STELLAR USD': 'XLM-USD',
               'OKB USD': 'OKB-USD', 'NEAR PROTOCOL USD': 'NEAR-USD', 'OPTIMISM USD': 'OP-USD',
               'INJECTIVE USD': 'INJ-USD', 'APTOS USD': 'APT21794-USD', 'MONERO USD': 'XMR-USD',
               'FILECOIN USD': 'FIL-USD', 'LIDO DAO USD': 'LDO-USD', 'CELESTIA USD': 'TIA22861-USD',
               'HEDERA USD': 'HBAR-USD', 'WRAPPED HBAR USD': 'WHBAR-USD', 'IMMUTABLE USD': 'IMX10603-USD',
               'WRAPPED EOS USD': 'WEOS-USD', 'ARBITRUM USD': 'ARB11841-USD', 'KASPA USD': 'KAS-USD',
               'BITCOIN BEP2 USD': 'BTCB-USD', 'STACKS USD': 'STX4847-USD', 'MANTLE USD': 'MNT27075-USD',
               'FIRST DIGITAL USD USD': 'FDUSD-USD', 'VECHAIN USD': 'VET-USD', 'CRONOS USD': 'CRO-USD',
               'WRAPPED BEACON ETH USD': 'WBETH-USD', 'TRUEUSD USD': 'TUSD-USD', 'SEI USD': 'SEI-USD',
               'MAKER USD': 'MKR-USD', 'HEX USD': 'HEX-USD', 'ROCKET POOL ETH USD': 'RETH-USD',
               'BITCOIN SV USD': 'BSV-USD', 'RENDER USD': 'RNDR-USD', 'BITTENSOR USD': 'TAO22974-USD',
               'THE GRAPH USD': 'GRT6719-USD', 'ALGORAND USD': 'ALGO-USD', 'ORDI USD': 'ORDI-USD',
               'AAVE USD': 'AAVE-USD', 'THORCHAIN USD': 'RUNE-USD', 'QUANT USD': 'QNT-USD',
               'MULTIVERSX USD': 'EGLD-USD', 'SUI USD': 'SUI20947-USD', 'MINA USD': 'MINA-USD',
               'SATS USD': '1000SATS-USD', 'FLOW USD': 'FLOW-USD', 'HELIUM USD': 'HNT-USD',
               'FANTOM USD': 'FTM-USD', 'SYNTHETIX USD': 'SNX-USD', 'THE SANDBOX USD': 'SAND-USD',
               'THETA NETWORK USD': 'THETA-USD', 'AXIE INFINITY USD': 'AXS-USD', 'TEZOS USD': 'XTZ-USD',
               'BEAM USD': 'BEAM28298-USD', 'BITTORRENT(NEW) USD': 'BTT-USD', 'KUCOIN TOKEN USD': 'KCS-USD',
               'DYDX (ETHDYDX) USD': 'ETHDYDX-USD', 'FTX TOKEN USD': 'FTT-USD', 'ASTAR USD': 'ASTR-USD',
               'WEMIX USD': 'WEMIX-USD', 'BLUR USD': 'BLUR-USD', 'CHEELEE USD': 'CHEEL-USD',
               'CHILIZ USD': 'CHZ-USD', 'BITGET TOKEN USD': 'BGB-USD', 'DECENTRALAND USD': 'MANA-USD',
               'NEO USD': 'NEO-USD', 'OSMOSIS USD': 'OSMO-USD', 'EOS USD': 'EOS-USD', 'BONK USD': 'BONK-USD',
               'KAVA USD': 'KAVA-USD', 'WOO USD': 'WOO-USD', 'KLAYTN USD': 'KLAY-USD', 'FLARE USD': 'FLR-USD',
               'OASIS NETWORK USD': 'ROSE-USD', 'IOTA USD': 'IOTA-USD', 'USDD USD': 'USDD-USD',
               'TERRA CLASSIC USD': 'LUNC-USD'}

commodities_dict = { "BRENT CRUDE OIL LAST DAY FINANC": "BZ=F", "COCOA": "CC=F", "COFFEE": "KC=F", "COPPER": "HG=F",
                    "CORN FUTURES": "ZC=F", "COTTON": "CT=F", "HEATING OIL": "HO=F", "KC HRW WHEAT FUTURES": "KE=F",
                    "LEAN HOGS FUTURES": "HE=F", "LIVE CATTLE FUTURES": "LE=F", "MONT BELVIEU LDH PROPANE (OPIS)": "B0=F",
                    "NATURAL GAS": "NG=F", "ORANGE JUICE": "OJ=F", "GOLD": "GC=F", "OAT FUTURES": "ZO=F",
                    "PALLADIUM": "PA=F", "CRUDE OIL": "CL=F", "PLATINUM": "PL=F", "RBOB GASOLINE": "RB=F",
                    "RANDOM LENGTH LUMBER FUTURES": "LBS=F", "ROUGH RICE FUTURES": "ZR=F", "SILVER": "SI=F",
                    "SOYBEAN FUTURES": "ZS=F", "SOYBEAN OIL FUTURES": "ZL=F", "S&P COMPOSITE 1500 ESG TILTED I": "ZM=F",
                    "SUGAR": "SB=F", "WISDOMTREE INTERNATIONAL HIGH D": "GF=F"
                }
b3_stocks =         {
                    "3M": "MMMC34.SA", "ABBOTT LABORATORIES": "ABTT34.SA", "AES BRASIL": "AESB3.SA", "AF INVEST": "AFHI11.SA",
                    "AFLUENTE T": "AFLT3.SA", "AGRIBRASIL": "GRAO3.SA", "AGROGALAXY": "AGXY3.SA", "ALIANSCESONAE": "ALSO3.SA",
                    "ALLIAR": "AALR3.SA", "ALPER": "APER3.SA", "GOOGLE": "GOGL35.SA", "ALUPAR INVESTIMENTO": "ALUP4.SA",
                    "AMC ENTERT H": "A2MC34.SA", "AMERICAN EXPRESS": "AXPB34.SA", "APPLE": "AAPL34.SA", "ARCELOR": "ARMT34.SA",
                    "ATT INC": "ATTB34.SA", "AUREN ENERGIA": "AURE3.SA", "AVALARA INC": "A2VL34.SA", "AVON": "AVON34.SA",
                    "BANCO DO BRASIL": "BBAS3.SA", "BANCO INTER": "BIDI3.SA", "BANCO MERCANTIL DE INVESTIMENTOS": "BMIN3.SA",
                    "BANCO PAN": "BPAN4.SA", "BANK AMERICA": "BOAC34.SA", "BANPARA": "BPAR3.SA", "BANRISUL": "BRSR3.SA",
                    "BATTISTELLA": "BTTL3.SA", "BAUMER": "BALM3.SA", "BB SEGURIDADE": "BBSE3.SA", "BEYOND MEAT": "B2YN34.SA",
                    "BIOMM": "BIOM3.SA", "BIOTOSCANA": "GBIO33.SA", "BMG": "BMGB4.SA", "BRASIL BROKERS": "BBRK3.SA",
                    "BRMALLS": "BRML3.SA", "BTG S&P 500 CI": "SPXB11.SA", "BTG SMLL CAPCI": "SMAB11.SA", "CAESARS ENTT": "C2ZR34.SA",
                    "CAIXA AGÊNCIAS": "CXAG11.SA", "CAMDEN PROP": "C2PT34.SA", "CAMIL": "CAML3.SA", "CARREFOUR": "CRFB3.SA",
                    "CARTESIA FIICI": "CACR11.SA", "CASAN": "CASN4.SA", "CEB": "CEBR6.SA", "CEEE-D": "CEED4.SA",
                    "CEEE-GT": "EEEL4.SA", "CEG": "CEGR3.SA", "CELESC": "CLSC4.SA", "CELPE": "CEPE6.SA", "CELULOSE IRANI": "RANI3.SA",
                    "CEMIG": "CMIG4.SA", "CESP": "CESP6.SA", "CHEVRON": "CHVX34.SA", "CHURCHILL DW": "C2HD34.SA", "CISCO": "CSCO34.SA",
                    "CITIGROUP": "CTGP34.SA", "CLEARSALE": "CLSA3.SA", "COCA-COLA": "COCA34.SA", "COELCE": "COCE6.SA",
                    "COINBASE GLOB": "C2OI34.SA", "COLGATE": "COLG34.SA", "COMGÁS": "CGAS3.SA", "CONOCOPHILLIPS": "COPH34.SA",
                    "COPEL UNT N2": "CPLE11.SA", "COPEL": "CPLE6.SA", "CPFL ENERGIA": "CPFE3.SA", "CSN": "CSNA3.SA",
                    "CSU CARDSYST": "CARD3.SA", "CYRUSONE INC": "C2ON34.SA", "DEXCO": "DXCO3.SA", "DEXXOS PART": "DEXP3.SA",
                    "DIMED": "PNVL3.SA", "DOMMO": "DMMO3.SA", "DOORDASH INC": "D2AS34.SA", "DRAFTKINGS": "D2KN34.SA",
                    "EBAY": "EBAY34.SA", "ENAUTA PART": "ENAT3.SA", "ENERGISA MT": "ENMT3.SA", "ENGIE BRASIL": "EGIE3.SA",
                    "EQI RECECI": "EQIR11.SA", "EUCATEX": "EUCA4.SA", "EXXON MOBIL": "EXXO34.SA", "FERBASA": "FESA4.SA",
                    "FIAGRO JGP CI": "JGPX11.SA", "FIAGRO RIZA CI": "RZAG11.SA", "FII BRIO ME CI": "BIME11.SA",
                    "FII CYRELA CI ES": "CYCR11.SA", "FII GTIS LG": "GTLG11.SA", "FII HUSI CI ES": "HUSI11.SA",
                    "FII JS A FINCI": "JSAF11.SA", "FII MORE CRICI ER": "MORC11.SA", "FII PLUR URBCI": "PURB11.SA",
                    "FII ROOFTOPICI": "ROOF11.SA", "FLEURY": "FLRY3.SA", "FREEPORT": "FCXO34.SA", "FT CLOUD CPT": "BKYY39.SA",
                    "FT DJ INTERN": "BFDN39.SA", "FT EQ OPPORT": "BFPX39.SA", "FT HCARE ALPH DRN": "BFXH39.SA",
                    "FT INTL EQ OP": "BFPI39.SA", "FT MOR DV LEA": "BFDL39.SA", "FT NASD CYBER": "BCIR39.SA",
                    "FT NASD100 EQ": "BQQW39.SA", "FT NASD100 TC": "BQTC39.SA", "FT NAT GAS": "BFCG39.SA",
                    "FT NYSE BIOT DRN": "BFBI39.SA", "FT RISI DIVID": "BFDA39.SA", "FT TECH ALPH": "BFTA39.SA",
                    "G2D INVESTMENTS": "G2DI33.SA", "GE": "GEOO34.SA", "GENERAL SHOPPING": "GSHP3.SA", "GERD PARANAPANEMA": "GEPA4.SA",
                    "GERDAU": "GOAU4.SA", "GETNET": "GETT11.SA", "GODADDY INC": "G2DD34.SA", "GOLDMAN SACHS": "GSGI34.SA",
                    "GRADIENTE": "IGBR3.SA", "HALLIBURTON": "HALI34.SA", "HONEYWELL": "HONB34.SA", "HP COMPANY": "HPQB34.SA",
                    "HYPERA PHARMA": "HYPE3.SA", "IBM": "IBMB34.SA", "IGUATEMI S.A.": "IGTI3.SA", "INFRACOMMERCE": "IFCM3.SA",
                    "INSTITUTO HERMES PARDINI SA": "PARD3.SA", "INTEL": "ITLC34.SA", "INVESTO ALUG": "ALUG11.SA",
                    "INVESTO USTK": "USTK11.SA", "INVESTO WRLD": "WRLD11.SA", "IRB BRASIL RE": "IRBR3.SA", "ISA CTEEP": "TRPL4.SA",
                    "ISHARES CSMO": "CSMO.SA", "ISHARES MILA": "MILA.SA", "ITAÚ UNIBANCO": "ITUB4.SA", "ITAÚSA": "ITSA4.SA",
                    "JBS": "JBSS3.SA", "JOHNSON": "JNJB34.SA", "JPMORGAN": "JPMC34.SA", "KINGSOFT CHL": "K2CG34.SA",
                    "KLABIN S/A": "KLBN11.SA", "LINX": "LINX3.SA", "LIVETECH": "LVTC3.SA", "LOCAWEB": "LWSA3.SA", "LOG": "LOGG3.SA",
                    "LPS BRASIL": "LPSB3.SA", "MARFRIG": "MRFG3.SA", "MASTERCARD": "MSCD34.SA", "MDIASBRANCO": "MDIA3.SA",
                    "MEDICAL P TR": "M2PW34.SA", "MERCANTIL DO BRASIL FINANCEIRA": "MERC4.SA", "MERCK": "MRCK34.SA",
                    "MICROSOFT": "MSFT34.SA", "MINERVA": "BEEF3.SA", "MMX MINERAÇÃO": "MMXM3.SA", "MORGAN STANLEY": "MSBR34.SA",
                    "MSCIGLMIVOLF": "BCWV39.SA", "MULTIPLAN": "MULT3.SA", "NATURA": "NTCO3.SA", "NEOENERGIA": "NEOE3.SA", "NU HOLDINGS": "NUBR33.SA",
                    "NU RENDA IBOV SMART DIVIDENDOS (NDIV11)": "NDIV11.SA", "ODONTOPREV": "ODPV3.SA","OI": "OIBR4.SA", "OMEGA ENERGIA": 
                    "MEGA3.SA", "ONCOCLÍNICAS": "ONCO3.SA","ORACLE": "ORCL34.SA", "OSX BRASIL": "OSXB3.SA", "OUROFINO S/A": "OFSA3.SA",
                    "PADTEC": "PDTC3.SA", "PÃO DE AÇÚCAR": "PCAR3.SA", "PARANAPANEMA": "PMAM3.SA","PEPSI": "PEPB34.SA", "PETROBRAS":"PETR4.SA", 
                    "PETRORECÔNCAVO GERAL SA": "RECV3.SA","PETRORIO": "PRIO3.SA", "PFIZER": "PFIZ34.SA", "PORTO SEGURO": "PSSA3.SA",
                    "PPLA": "PPLA11.SA", "PRIVALIA": "PRVA3.SA", "PROCTER GAMBLE": "PGCO34.SA", "PROCTOR GAMBLE": "PGCO34.SA", "QUALCOMM": "QCOM34.SA", 
                    "QUALICORP": "QUAL3.SA","RAD": "RADL3.SA", "RENOVA": "RNEW4.SA", "RIO BRAVO": "RBIV11.SA", "SABESP": "SBSP3.SA",
                    "SANEPAR": "SAPR4.SA", "SANTANDER BR": "SANB11.SA", "SÃO CARLOS": "SCAR3.SA", "SÃO MARTINHO": "SMTO3.SA", 
                    "SCHLUMBERGER": "SLBG34.SA", "SEA LTD": "S2EA34.SA","SHOPIFY INC": "S2HO34.SA", "SMART FIT": "SMFT3.SA", 
                    "SNOWFLAKE": "S2NW34.SA", "SP500 VALUE": "BIVE39.SA", "SP500GROWTH": "BIVW39.SA", "SQUARE INC": "S2QU34.SA",
                    "SQUARESPACE": "S2QS34.SA", "STARBUCKS": "SBUB34.SA", "STONE CO": "STOC31.SA","STORE CAPITAL": "S2TO34.SA", "SUN COMMUN": "S2UI34.SA", "SUZANO HOLDING": "NEMO6.SA",
                     "SUZANO PAPEL": "SUZB3.SA", "SYN PROP TECH": "SYNE3.SA", "TAEsa": "TAEE11.SA","TELADOCHEALT": "T2DH34.SA", "TELEBRAS": "TELB4.SA", "TELEFÔNICA BRASIL S.A": "VIVT3.SA",
                    "TELLUS DESENVOLVIMENTO LOGÍSTICO": "TELD11.SA", "TERRA SANTA AGRO SA": "LAND3.SA", "TIM PARTICIPAÇÕES": "TIMS3.SA", "TOTVS": "TOTS3.SA", "TRADE DESK": "T2TD34.SA",
                    "TRADERSCLUB": "TRAD3.SA", "TRONOX": "CRPG6.SA", "UIPATH INC": "P2AT34.SA","ULTRAPAR": "UGPA3.SA", "UNIPAR": "UNIP6.SA", "UNITY SOFTWR": "U2ST34.SA",
                    "VALE": "VALE3.SA", "VERIZON": "VERZ34.SA", "VISA": "VISA34.SA", "VIVARA": "VIVA3.SA", "VIVEO": "VVEO3.SA", "VOTORANTIM ASSET MANAGEMENT": "VSEC11.SA", "WALMART": "WALM34.SA",
                    "WELLS FARGO": "WFCO34.SA", "WEST PHARMA": "W2ST34.SA", "WILSON SONS": "PORT3.SA", "XEROX": "XRXB34.SA", "XP INC": "XPBR31.SA", "ZYNGA INC": "Z2NG34.SA"
                    }

indexes_dict =     {
                    'S&P GSCI': 'GD=F', 'IBOVESPA': '^BVSP', 'S&P/CLX IPSA': '^IPSA',
                    'MERVAL': '^MERV', 'IPC MEXICO': '^MXX', 'S&P 500': '^GSPC',
                    'Dow Jones Industrial Average': '^DJI', 'NASDAQ Composite': '^IXIC',
                    'NYSE COMPOSITE (DJ)': '^NYA', 'NYSE AMEX COMPOSITE INDEX': '^XAX',
                    'Russell 2000': '^RUT', 'CBOE Volatility Index': '^VIX',
                    'S&P/TSX Composite index': '^GSPTSE', 'FTSE 100': '^FTSE',
                    'DAX PERFORMANCE-INDEX': '^GDAXI', 'CAC 40': '^FCHI',
                    'ESTX 50 PR.EUR': '^STOXX50E', 'Euronext 100 Index': '^N100',
                    'BEL 20': '^BFX', 'MOEX Russia Index': 'IMOEX.ME', 'Nikkei 225': '^N225',
                    'HANG SENG INDEX': '^HSI', 'SSE Composite Index': '000001.SS',
                    'Shenzhen Index': '399001.SZ', 'STI Index': '^STI', 'S&P/ASX 200': '^AXJO',
                    'ALL ORDINARIES': '^AORD', 'S&P BSE SENSEX': '^BSESN', 'IDX COMPOSITE': '^JKSE',
                    'FTSE Bursa Malaysia KLCI': '^KLSE', 'S&P/NZX 50 INDEX GROSS ( GROSS': '^NZ50',
                    'KOSPI Composite Index': '^KS11', 'TSEC weighted index': '^TWII',
                    'TA-125': '^TA125.TA', 'Top 40 USD Net TRI Index': '^JN0U.JO', 'NIFTY 50': '^NSEI'
                    }

sp500_dict = {
    '3M': 'MMM', 'A. O. SMITH': 'AOS', 'ABBOTT': 'ABT', 'ABBVIE': 'ABBV', 'ACCENTURE': 'ACN', 'ADOBE INC.': 'ADBE',
    'ADVANCED MICRO DEVICES': 'AMD', 'AES CORPORATION': 'AES', 'AFLAC': 'AFL', 'AGILENT TECHNOLOGIES': 'A', 'AIR PRODUCTS AND CHEMICALS': 'APD',
    'AIRBNB': 'ABNB', 'AKAMAI': 'AKAM', 'ALBEMARLE CORPORATION': 'ALB', 'ALEXANDRIA REAL ESTATE EQUITIES': 'ARE', 'ALIGN TECHNOLOGY': 'ALGN',
    'ALLEGION': 'ALLE', 'ALLIANT ENERGY': 'LNT', 'ALLSTATE': 'ALL', 'GOOGLE': 'GOOGL', 'GOOGLE': 'GOOG',
    'ALTRIA': 'MO', 'AMAZON': 'AMZN', 'AMCOR': 'AMCR', 'AMEREN': 'AEE', 'AMERICAN AIRLINES GROUP': 'AAL', 'AMERICAN ELECTRIC POWER': 'AEP',
    'AMERICAN EXPRESS': 'AXP', 'AMERICAN INTERNATIONAL GROUP': 'AIG', 'AMERICAN TOWER': 'AMT', 'AMERICAN WATER WORKS': 'AWK', 'AMERIPRISE FINANCIAL': 'AMP',
    'AMETEK': 'AME', 'AMGEN': 'AMGN', 'AMPHENOL': 'APH', 'ANALOG DEVICES': 'ADI', 'ANSYS': 'ANSS', 'AON': 'AON',
    'APA CORPORATION': 'APA', 'APPLE INC.': 'AAPL', 'APPLIED MATERIALS': 'AMAT', 'APTIV': 'APTV', 'ARCH CAPITAL GROUP': 'ACGL', 'ARCHER-DANIELS-MIDLAND': 'ADM',
    'ARISTA NETWORKS': 'ANET', 'ARTHUR J. GALLAGHER & CO.': 'AJG', 'ASSURANT': 'AIZ', 'AT&T': 'T', 'ATMOS ENERGY': 'ATO', 'AUTODESK': 'ADSK',
    'AUTOMATED DATA PROCESSING': 'ADP', 'AUTOZONE': 'AZO', 'AVALONBAY COMMUNITIES': 'AVB', 'AVERY DENNISON': 'AVY', 'AXON ENTERPRISE': 'AXON', 'BAKER HUGHES': 'BKR',
    'BALL CORPORATION': 'BALL', 'BANK OF AMERICA': 'BAC', 'BANK OF NEW YORK MELLON': 'BK', 'BATH & BODY WORKS, INC.': 'BBWI', 'BAXTER INTERNATIONAL': 'BAX', 'BECTON DICKINSON': 'BDX',
    'BERKSHIRE HATHAWAY': 'BRK.B', 'BEST BUY': 'BBY', 'BIO-RAD': 'BIO', 'BIO-TECHNE': 'TECH', 'BIOGEN': 'BIIB', 'BLACKROCK': 'BLK', 'BLACKSTONE': 'BX',
    'BOEING': 'BA', 'BOOKING HOLDINGS': 'BKNG', 'BORGWARNER': 'BWA', 'BOSTON PROPERTIES': 'BXP', 'BOSTON SCIENTIFIC': 'BSX', 'BRISTOL MYERS SQUIBB': 'BMY', 'BROADCOM INC.': 'AVGO',
    'BROADRIDGE FINANCIAL SOLUTIONS': 'BR', 'BROWN & BROWN': 'BRO', 'BROWN–FORMAN': 'BF.B', 'BUILDERS FIRSTSOURCE': 'BLDR', 'BUNGE GLOBAL SA': 'BG', 'CADENCE DESIGN SYSTEMS': 'CDNS',
    'CAESARS ENTERTAINMENT': 'CZR', 'CAMDEN PROPERTY TRUST': 'CPT', 'CAMPBELL SOUP COMPANY': 'CPB', 'CAPITAL ONE': 'COF', 'CARDINAL HEALTH': 'CAH', 'CARMAX': 'KMX',
    'CARNIVAL': 'CCL', 'CARRIER GLOBAL': 'CARR', 'CATALENT': 'CTLT', 'CATERPILLAR INC.': 'CAT', 'CBOE GLOBAL MARKETS': 'CBOE', 'CBRE GROUP': 'CBRE', 'CDW': 'CDW',
    'CELANESE': 'CE', 'CENCORA': 'COR', 'CENTENE CORPORATION': 'CNC', 'CENTERPOINT ENERGY': 'CNP', 'CERIDIAN': 'CDAY', 'CF INDUSTRIES': 'CF', 'CH ROBINSON': 'CHRW',
    'CHARLES RIVER LABORATORIES': 'CRL', 'CHARLES SCHWAB CORPORATION': 'SCHW', 'CHARTER COMMUNICATIONS': 'CHTR', 'CHEVRON CORPORATION': 'CVX', 'CHIPOTLE MEXICAN GRILL': 'CMG',
    'CHUBB LIMITED': 'CB', 'CHURCH & DWIGHT': 'CHD', 'CIGNA': 'CI', 'CINCINNATI FINANCIAL': 'CINF', 'CINTAS': 'CTAS', 'CISCO': 'CSCO', 'CITIGROUP': 'C',
    'CITIZENS FINANCIAL GROUP': 'CFG', 'CLOROX': 'CLX', 'CME GROUP': 'CME', 'CMS ENERGY': 'CMS', 'COCA-COLA COMPANY (THE)': 'KO', 'COGNIZANT': 'CTSH', 'COLGATE-PALMOLIVE': 'CL',
    'COMCAST': 'CMCSA', 'COMERICA': 'CMA', 'CONAGRA BRANDS': 'CAG', 'CONOCOPHILLIPS': 'COP', 'CONSOLIDATED EDISON': 'ED', 'CONSTELLATION BRANDS': 'STZ', 'CONSTELLATION ENERGY': 'CEG',
    'COOPERCOMPANIES': 'COO', 'COPART': 'CPRT', 'CORNING INC.': 'GLW', 'CORTEVA': 'CTVA', 'COSTAR GROUP': 'CSGP', 'COSTCO': 'COST', 'COTERRA': 'CTRA', 'CROWN CASTLE': 'CCI',
    'CSX': 'CSX', 'CUMMINS': 'CMI', 'CVS HEALTH': 'CVS', 'DANAHER CORPORATION': 'DHR', 'DARDEN RESTAURANTS': 'DRI', 'DAVITA INC.': 'DVA', 'JOHN DEERE': 'DE', 'DELTA AIR LINES': 'DAL',
    'DENTSPLY SIRONA': 'XRAY', 'DEVON ENERGY': 'DVN', 'DEXCOM': 'DXCM', 'DIAMONDBACK ENERGY': 'FANG', 'DIGITAL REALTY': 'DLR', 'DISCOVER FINANCIAL': 'DFS', 'DOLLAR GENERAL': 'DG',
    'DOLLAR TREE': 'DLTR', 'DOMINION ENERGY': 'D', 'DOMINO\'S': 'DPZ', 'DOVER CORPORATION': 'DOV', 'DOW INC.': 'DOW', 'DR HORTON': 'DHI', 'DTE ENERGY': 'DTE', 'DUKE ENERGY': 'DUK',
    'DUPONT': 'DD', 'EASTMAN CHEMICAL COMPANY': 'EMN', 'EATON CORPORATION': 'ETN', 'EBAY': 'EBAY', 'ECOLAB': 'ECL', 'EDISON INTERNATIONAL': 'EIX', 'EDWARDS LIFESCIENCES': 'EW',
    'ELECTRONIC ARTS': 'EA', 'ELEVANCE HEALTH': 'ELV', 'ELI LILLY AND COMPANY': 'LLY', 'EMERSON ELECTRIC': 'EMR', 'ENPHASE': 'ENPH', 'ENTERGY': 'ETR', 'EOG RESOURCES': 'EOG',
    'EPAM SYSTEMS': 'EPAM', 'EQT': 'EQT', 'EQUIFAX': 'EFX', 'EQUINIX': 'EQIX', 'EQUITY RESIDENTIAL': 'EQR', 'ESSEX PROPERTY TRUST': 'ESS', 'ESTÉE LAUDER COMPANIES (THE)': 'EL',
    'ETSY': 'ETSY', 'EVEREST RE': 'EG', 'EVERGY': 'EVRG', 'EVERSOURCE': 'ES', 'EXELON': 'EXC', 'EXPEDIA GROUP': 'EXPE', 'EXPEDITORS INTERNATIONAL': 'EXPD', 'EXTRA SPACE STORAGE': 'EXR',
    'EXXONMOBIL': 'XOM', 'F5, INC.': 'FFIV', 'FACTSET': 'FDS', 'FAIR ISAAC': 'FICO', 'FASTENAL': 'FAST', 'FEDERAL REALTY': 'FRT', 'FEDEX': 'FDX', 'FIDELITY NATIONAL INFORMATION SERVICES': 'FIS',
    'FIFTH THIRD BANK': 'FITB', 'FIRST SOLAR': 'FSLR', 'FIRSTENERGY': 'FE', 'FISERV': 'FI', 'FLEETCOR': 'FLT', 'FMC CORPORATION': 'FMC', 'FORD MOTOR COMPANY': 'F', 'FORTINET': 'FTNT',
    'FORTIVE': 'FTV', 'FOX CORPORATION (CLASS A)': 'FOXA', 'FOX CORPORATION (CLASS B)': 'FOX', 'FRANKLIN TEMPLETON': 'BEN', 'FREEPORT-MCMORAN': 'FCX', 'GARMIN': 'GRMN', 'GARTNER': 'IT',
    'GE HEALTHCARE': 'GEHC', 'GEN DIGITAL': 'GEN', 'GENERAC': 'GNRC', 'GENERAL DYNAMICS': 'GD', 'GENERAL ELECTRIC': 'GE', 'GENERAL MILLS': 'GIS', 'GENERAL MOTORS': 'GM', 'GENUINE PARTS COMPANY': 'GPC',
    'GILEAD SCIENCES': 'GILD', 'GLOBAL PAYMENTS': 'GPN', 'GLOBE LIFE': 'GL', 'GOLDMAN SACHS': 'GS', 'HALLIBURTON': 'HAL', 'HARTFORD (THE)': 'HIG', 'HASBRO': 'HAS', 'HCA HEALTHCARE': 'HCA',
    'HEALTHPEAK': 'PEAK', 'HENRY SCHEIN': 'HSIC', 'HERSHEY\'S': 'HSY', 'HESS CORPORATION': 'HES', 'HEWLETT PACKARD ENTERPRISE': 'HPE', 'HILTON WORLDWIDE': 'HLT', 'HOLOGIC': 'HOLX',
    'HOME DEPOT (THE)': 'HD', 'HONEYWELL': 'HON', 'HORMEL FOODS': 'HRL', 'HOST HOTELS & RESORTS': 'HST', 'HOWMET AEROSPACE': 'HWM', 'HP INC.': 'HPQ', 'HUBBELL INCORPORATED': 'HUBB',
    'HUMANA': 'HUM', 'HUNTINGTON BANCSHARES': 'HBAN', 'HUNTINGTON INGALLS INDUSTRIES': 'HII', 'IBM': 'IBM', 'IDEX CORPORATION': 'IEX', 'IDEXX LABORATORIES': 'IDXX',
    'ILLINOIS TOOL WORKS': 'ITW', 'ILLUMINA': 'ILMN', 'INCYTE': 'INCY', 'INGERSOLL RAND': 'IR', 'INSULET': 'PODD', 'INTEL': 'INTC', 'INTERCONTINENTAL EXCHANGE': 'ICE',
    'INTERNATIONAL FLAVORS & FRAGRANCES': 'IFF', 'INTERNATIONAL PAPER': 'IP', 'INTERPUBLIC GROUP OF COMPANIES (THE)': 'IPG', 'INTUIT': 'INTU', 'INTUITIVE SURGICAL': 'ISRG',
    'INVESCO': 'IVZ', 'INVITATION HOMES': 'INVH', 'IQVIA': 'IQV', 'IRON MOUNTAIN': 'IRM', 'J.B. HUNT': 'JBHT', 'JABIL': 'JBL', 'JACK HENRY & ASSOCIATES': 'JKHY', 'JACOBS SOLUTIONS': 'J',
    'JOHNSON & JOHNSON': 'JNJ', 'JOHNSON CONTROLS': 'JCI', 'JPMORGAN CHASE': 'JPM', 'JUNIPER NETWORKS': 'JNPR', 'KELLANOVA': 'K', 'KENVUE': 'KVUE', 'KEURIG DR PEPPER': 'KDP',
    'KEYCORP': 'KEY', 'KEYSIGHT': 'KEYS', 'KIMBERLY-CLARK': 'KMB', 'KIMCO REALTY': 'KIM', 'KINDER MORGAN': 'KMI', 'KLA CORPORATION': 'KLAC', 'KRAFT HEINZ': 'KHC', 'KROGER': 'KR',
    'L3HARRIS': 'LHX'
}

nasdaq_dict = {
    'Adobe Inc.': 'ADBE', 'ADP': 'ADP', 'Airbnb': 'ABNB', 'GOOGLE': 'GOOGL', 'GOOGLE': 'GOOG', 'Amazon': 'AMZN',
    'Advanced Micro Devices Inc.': 'AMD', 'American Electric Power': 'AEP', 'Amgen': 'AMGN', 'Analog Devices': 'ADI', 'Ansys': 'ANSS', 'Apple Inc.': 'AAPL',
    'Applied Materials': 'AMAT', 'ASML Holding': 'ASML', 'AstraZeneca': 'AZN', 'Atlassian': 'TEAM', 'Autodesk': 'ADSK', 'Baker Hughes': 'BKR',
    'Biogen': 'BIIB', 'Booking Holdings': 'BKNG', 'Broadcom Inc.': 'AVGO', 'Cadence Design Systems': 'CDNS', 'CDW Corporation': 'CDW',
    'Charter Communications': 'CHTR', 'Cintas': 'CTAS', 'Cisco': 'CSCO', 'Coca-Cola Europacific Partners': 'CCEP', 'Cognizant': 'CTSH', 'Comcast': 'CMCSA',
    'Constellation Energy': 'CEG', 'Copart': 'CPRT', 'CoStar Group': 'CSGP', 'Costco': 'COST', 'CrowdStrike': 'CRWD', 'CSX Corporation': 'CSX',
    'Datadog': 'DDOG', 'DexCom': 'DXCM', 'Diamondback Energy': 'FANG', 'Dollar Tree': 'DLTR', 'DoorDash': 'DASH', 'Electronic Arts': 'EA',
    'Exelon': 'EXC', 'Fastenal': 'FAST', 'Fortinet': 'FTNT', 'GE HealthCare': 'GEHC', 'Gilead Sciences': 'GILD', 'GlobalFoundries': 'GFS',
    'Honeywell': 'HON', 'Idexx Laboratories': 'IDXX', 'Illumina, Inc.': 'ILMN', 'Intel': 'INTC', 'Intuit': 'INTU', 'Intuitive Surgical': 'ISRG',
    'Keurig Dr Pepper': 'KDP', 'KLA Corporation': 'KLAC', 'Kraft Heinz': 'KHC', 'Lam Research': 'LRCX', 'Lululemon': 'LULU', 'Marriott International': 'MAR',
    'Marvell Technology': 'MRVL', 'MercadoLibre': 'MELI', 'Meta Platforms': 'META', 'Microchip Technology': 'MCHP', 'Micron Technology': 'MU', 'Microsoft': 'MSFT',
    'Moderna': 'MRNA', 'Mondelēz International': 'MDLZ', 'MongoDB Inc.': 'MDB', 'Monster Beverage': 'MNST', 'Netflix': 'NFLX', 'Nvidia': 'NVDA', 'NXP': 'NXPI',
    'O\'Reilly Automotive': 'ORLY', 'Old Dominion Freight Line': 'ODFL', 'Onsemi': 'ON', 'Paccar': 'PCAR', 'Palo Alto Networks': 'PANW', 'Paychex': 'PAYX',
    'PayPal': 'PYPL', 'PDD Holdings': 'PDD', 'PepsiCo': 'PEP', 'Qualcomm': 'QCOM', 'Regeneron': 'REGN', 'Roper Technologies': 'ROP', 'Ross Stores': 'ROST',
    'Sirius XM': 'SIRI', 'Splunk': 'SPLK', 'Starbucks': 'SBUX', 'Synopsys': 'SNPS', 'Take-Two Interactive': 'TTWO', 'T-Mobile US': 'TMUS', 'Tesla, Inc.': 'TSLA',
    'Texas Instruments': 'TXN', 'The Trade Desk': 'TTD', 'Verisk': 'VRSK', 'Vertex Pharmaceuticals': 'VRTX', 'Walgreens Boots Alliance': 'WBA',
    'Warner Bros. Discovery': 'WBD', 'Workday, Inc.': 'WDAY', 'Xcel Energy': 'XEL', 'Zscaler': 'ZS',
}

commodities_dict = {
    "Brent Crude Oil": "BZ=F", "Cocoa": "CC=F", "Coffee": "KC=F", "Copper": "HG=F", 
    "Corn Futures": "ZC=F", "Cotton": "CT=F", "Heating Oil": "HO=F", "KC HRW Wheat Futures": "KE=F", 
    "Lean Hogs Futures": "HE=F", "Live Cattle Futures": "LE=F", "Mont Belvieu LDH Propane (OPIS)": "B0=F", 
    "Natural Gas": "NG=F", "Orange Juice": "OJ=F", "OURO": "GC=F", "Oat Futures": "ZO=F", 
    "Palladium": "PA=F", "PETROLEO CRU": "CL=F", "Platinum": "PL=F", "RBOB Gasoline": "RB=F", 
    "Random Length Lumber Futures": "LBS=F", "Rough Rice Futures": "ZR=F", "Silver": "SI=F", 
    "Soybean Futures": "ZS=F", "Soybean Oil Futures": "ZL=F", "S&P Composite 1500 ESG Tilted I": "ZM=F", 
    "Sugar #11": "SB=F", "WisdomTree International High D": "GF=F"
}



selected_timeframes = st.selectbox('Select Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'], index=7)

assets_list = {'commodities':commodities_dict, 
               'b3_stocs': b3_stocks, '
               'SP500': sp500_dict,
               'NASDAC100':nasdaq_dict ,
               'indexes': indexes_dict,
               'currencies': currencies_dict, 
               'crypto': crypto_dict}


selected_dict_names = st.multiselect('Select dictionaries to combine', list(assets_list.keys()))
combined_dict = {}
for name in selected_dict_names:
    dictionary = assets_list.get(name)
    if dictionary:
        combined_dict.update(dictionary)

selected_ticker_dict = {}

if selected_dict_names:
    tickers = st.multiselect('Asset Selection', list(combined_dict.keys()))
    if tickers and st.button("Download data"):
        for key in tickers:
            if key in combined_dict:
                selected_ticker_dict[key] = combined_dict[key]
        session_state.data = download_data(selected_ticker_dict, selected_timeframes)

# Handle tickers entered manually
type_tickers = st.text_input('Enter Tickers (comma-separated):')
if type_tickers and st.button("Download data"):
    tickers = [ticker.strip() for ticker in type_tickers.split(',')]
    session_state.data = download_data(tickers, selected_timeframes)
    #session_state.data = asset_mapping(session_state.data, assets_list)

frequency = {
        'Daily': 'D',
        'Weekly': 'W',
        'Quaterly': '2W',
        'Monthly': 'M',
        'Bimonthly': '2M',
        'Quarterly': '3M',
        'Four-monthly': '4M',
        'Semiannual': '6M',
        'Annual': 'A'
    }	

# sidebar resampling function
st.sidebar.markdown('**Time Series Resampler**')
freq = st.sidebar.selectbox("Freq to resample:", list(frequency.keys()))
agg = st.sidebar.selectbox("Aggregation:", ['sum', 'mean', 'median', 'valor_exato'])
resample = st.sidebar.button("Resample dataframe")
if resample:
	if session_state.data is not None:
		session_state.data = date_resample(session_state.data,frequency[freq],agg)

# moving average for NaN ocurrencies
st.sidebar.markdown('**Moving Avarage**')
moving_avg_days =  st.sidebar.number_input('Day(s):',1, 100, 3,step=1)                     
method = st.sidebar.selectbox("Method:", ['gap', 'rolling'])

if st.sidebar.button("Apply") and session_state.data is not None:
	session_state.data = fill_moving_avg(session_state.data, moving_avg_days, method)       

st.sidebar.markdown('**Missing Values**')    
remove_nan = st.sidebar.button('Dropna')
if remove_nan:
    session_state.data = drop_nan_rows(session_state.data)

if session_state.data is not None:
    st.markdown(f'**Total of missing entries:** {session_state.data.isna().sum().sum()}')
    st.dataframe(session_state.data.isna().sum().to_frame().T)
    st.dataframe(session_state.data)
    tickers = [str(col).split("_")[0] for col in session_state.data.columns]
    tickers  = set(tickers)

st.subheader('Assets allocation', divider='rainbow')
with st.expander("See explanation"):
    st.markdown("""
            <div style="text-align: justify">
            
            1. **Enter Available Cash:** Start by entering the amount of cash you currently have available for investment. This helps us understand how much you have to work with.<br><br>
            
            2. **Enter Invested Cash:** If you already have some cash invested, enter that amount here. This allows us to consider your existing investments prior to optimization.<br><br>
            
            3. **Allocate Your Cash:** Once you have provided your financial details, you can allocate your cash to different assets represented by tickers (like stock symbols). For each asset, you can specify the percentage of your available cash you want to invest. Simply enter the percentage of shares you want to allocate for each asset.<br><br>
            
            4. **Review Your Allocation:** As you allocate your cash, we'll keep track of the total percentage of shares allocated across all assets. If the total reaches 100%, your allocation is complete, and you will be able to run optimization.
          
            </div>
        """, unsafe_allow_html=True)

resampling_options = ['A', 'AS', 'BA', 'BAS', '3M', '4M', '6M', '12M', 
                      'Q', 'BQ', 'QS', 'BQS', 
                      'M', 'BM', 'MS', 'BMS', 
                      'W', 'D'] 

total_shares = []
invested_cash = st.number_input("Enter invested cash", min_value=0.0, max_value=1e12, step=1000.0, value=100000.00, format="%.2f")

if session_state.data is not None:
    try:
        if 'tickers' in globals() and tickers is not None:
            for ticker in tickers:
                share = st.number_input(f'{ticker} share', min_value=0.0, max_value=1.0, value = 1.0/len(tickers), step=0.05, format="%.2f")
                total_shares.append(share)
                allocated_shares =  sum(total_shares)
                shares_to_allocate = 1 - allocated_shares
                
            if allocated_shares == 1.0:
                st.write(f'Allocation: {sum(total_shares) * 100:.2f}%')
                session_state.df = compute_investments(session_state.data, tickers, total_shares, invested_cash) 
		    
            elif 0.0 < allocated_shares < 1.0:
                st.write(f'You must allocate another {(shares_to_allocate * 100):.2f}% on assets!')
            
            else:
                st.write(f'Max Allocation exceeded. Please reshare {abs(shares_to_allocate * 100):.2f}%')
    except NameError:
        st.write("Please download tickers before continuing.")
        
if session_state.df is not None:
   st.dataframe(session_state.df)
   st.subheader('Optimization', divider='rainbow')
   close_price_data = [col for col  in session_state.data.columns if col.endswith('_Close')]
   session_state.portfolio  = session_state.data[close_price_data]

if session_state.portfolio is not None and not session_state.portfolio.empty:
    trading_days = st.number_input(f'Please Select timeframe for returns', min_value=1, max_value=365, step=1, value = 252)
    resample_list = st.selectbox('Select resampling frequency:', options=resampling_options, index=resampling_options.index('A'))
    risk_free_rate = st.number_input(f'Please Select risk free rate', min_value=0.0, max_value=1.0, step=0.03, value = 0.05)
    risk_taken = st.number_input(f'Please Select anualized risk of investment:', min_value=0.0, max_value=1.0, step=0.02, value = 0.1)
    expected_return = st.number_input(f'Please Select anualized expected returns', min_value=0.0, max_value=1.0, step=0.05, value = 0.15)
    simulations = st.number_input(f'Please Select number of simulations', min_value=100, max_value=100000, step=50, value = 1000)
    expected_sharpe = (expected_return - risk_free_rate)/ risk_taken
    st.markdown(f'Expected Sharpe Ratio: **{expected_sharpe:.2f}**')
    run_simulations = st.button('Run simulations')
    if run_simulations:
        simulated_portfolios = efficient_frontier(session_state.portfolio, trading_days, total_shares, risk_free_rate, simulations, resample_list)
        plot_efficient_frontier(simulated_portfolios, risk_free_rate, expected_sharpe,expected_return, risk_taken)
        

if session_state.portfolio is not None and session_state.portfolio.shape[1] >= 2:
    st.subheader('Backtesting Strategy', divider='rainbow')
    first_sharpe = st.number_input('Please select number of days to obtain the first  round of weights:', min_value=1, max_value=10000, step=15, value=252)
    offset = st.number_input('Please select number of days to jump:', min_value=1, max_value=10000, step=15, value=30)
    
    dates_range = session_state.portfolio.index.unique()
    backtest_dfs = []
    first_df =  session_state.portfolio.iloc[:first_sharpe]
    first_df['ID'] = 1
    first_df['date'] = session_state.portfolio.index[first_sharpe]  # Defina a data para a última data do intervalo
    df_id = 2
    for i in range(first_sharpe, len(dates_range), offset):
        starting_date = dates_range[i] 
        offset_date = dates_range[min(i + offset - 1, len(dates_range) - 1)]  # Garante que offset_date não ultrapasse o final de dates_range
        split_df = session_state.portfolio.loc[starting_date:offset_date]
        split_df['ID'] = df_id
        split_df['date'] = offset_date
        backtest_dfs.append(split_df)
        df_id += 1
        starting_date = offset_date
    
    
    min_constraints = [0.0] * len(tickers)
    max_constraints = [1.0] * len(tickers)

    with st.expander("***Optional Constraints***"):
        st.markdown("""
            1. Choose min and max constraints for the weights of your assets.
            
            2. The selected constraints will be normalized so that weights sum 1. Therefore, you may see weights different from 
            the constraints you selected, however, they should be proportional. 
        
        """)

        for num in range(len(tickers)):
            min_constraint = st.number_input(f'Select min weight constraint for asset {list(tickers)[num]}', min_value=0.0, max_value=1.0, step=0.05, value = 0.0)
            max_constraint = st.number_input(f'Select max weight constraint for asset {list(tickers)[num]}', min_value=0.0, max_value=1.0, step=0.05, value = 1.0)
            min_constraints[num] = min_constraint
            max_constraints[num] = max_constraint
 
backtesting = st.button('backtest')
if backtesting:
    optimized_dfs = backtest_frontier(backtest_dfs, risk_free_rate, trading_days)
    portfolios_anazlyed = len(optimized_dfs)
    backtested_df = get_max_sharpe_per_id(optimized_dfs, max_constraints, min_constraints)
    backtested_df.set_index('Date', inplace=True, drop=True)
    weight_columns = [col for col in backtested_df.columns if col.endswith('_Weight')]
    aux = session_state.data.copy()
    aux.columns = [col.replace('_Close', '_Price') for col in aux.columns] 
    price_columns = [col for col in aux.columns if col.endswith('_Price')]	
    backtested_df = backtested_df.merge(aux[price_columns], left_index=True, right_index=True, how='left')
    roi_df = returns_till_date(backtested_df, invested_cash, price_columns, weight_columns)
    
    data = {'ID': [] ,'date': [], 'asset': [], 'price': [], 'weight': [], 'quantity': []}
    for date, row in backtested_df.iterrows():
        for price_col, weight_col in zip(price_columns, weight_columns):
            asset = price_col.replace('_Price', '')
            data['ID'].append(row['ID'])
            data['date'].append(date)
            data['asset'].append(asset)
            data['price'].append(row[price_col])
            data['weight'].append(row[weight_col])
            data['quantity'].append(invested_cash*row[weight_col] /row[price_col])
        
    optimize_df = pd.DataFrame(data)

    dfs_by_id = {}
    for unique_id in optimize_df['ID'].unique():
        df_id = optimize_df[optimize_df['ID'] == unique_id][['ID', 'asset', 'price', 'weight']]
        dfs_by_id[unique_id] = df_id

    # Combine DataFrames by ID
    combined_dfs = []
    results = []
    unique_ids = sorted(dfs_by_id.keys())
    for i in range(len(unique_ids) - 1):
        current_id = unique_ids[i]
        next_id = unique_ids[i + 1]
        df_t1 = dfs_by_id[current_id].copy()
        df_t1.columns = ['ID', 'asset', f'pricesT1', f'weightsT1']
        
        df_t2 = dfs_by_id[next_id].copy()
        df_t2.columns = ['ID', 'asset', f'pricesT2', f'weightsT2']
        
        combined_df = pd.merge(df_t1, df_t2, on='asset', suffixes=('_t1', '_t2'))
        combined_df['qtT1'] = invested_cash * combined_df['weightsT1'] / combined_df['pricesT1']
        combined_dfs.append(combined_df)
    
    st.markdown(f'***Number of portfolios analyzed:*** {portfolios_anazlyed}')   
    st.dataframe(backtested_df) 
    st.markdown(f'***Number of optimizations:*** {len(combined_dfs)}')
    st.markdown(f"**Return till date:** {roi_df['return_till_date'][1]:.3f}")
    optimizedDf = optimizeBySharpe(combined_dfs[0])
    st.markdown(f"**Return till date:** {roi_df['return_till_date'][1]:.3f}")
    results.append(optimizedDf)
    for i in range(1, len(combined_dfs)):
        lastOptimized = results[i - 1]
        combined_dfs[i]['qtT1'] = lastOptimized['optQtT2']
        combined_dfs[i]['weightsT1'] = lastOptimized['optWeightsT2']
        st.markdown(f"**Return till date:** {roi_df['return_till_date'][i+1]:.3f}")
        optimizedDf = optimizeBySharpe(combined_dfs[i])
        results.append(optimizedDf)

if session_state.df is not None or session_state.data is not None or session_state.portfolio is not None or session_state.backtest is not None or session_state.optimized_data is not None:
    st.subheader("Downloads:", divider='rainbow')
    mapping = {'assets': 'data', 'allocation': 'df', 'portfolio': 'portfolio', 'backtest': 'backtest', 'optimized_data': 'optimized_data'}
    download_option = st.selectbox("Select for Download:", list(mapping.keys()))
    if download_option in mapping:
        download_button = st.button('Download')
        if download_button:
            download_link = download_dfs(session_state, mapping[download_option], mapping)
            if download_link:
                st.markdown(download_link, unsafe_allow_html=True)
    else:
        st.error("Opção de download inválida.")

def generate_combinations(dictionary, assets_per_portfolio, combinations_limit=10000):
    assets = list(dictionary.values())
    combinations = []
    for combination in itertools.combinations(assets, assets_per_portfolio):
        combinations.append(combination)
        if len(combinations) >= combinations_limit:
            break
    return combinations

def download_portfolios(data, period='5y'):
    dfs = {}
    tickers_with_errors = []
    
    if isinstance(data, dict):
        for name, ticker in data.items():
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period)
                hist.columns = [f"{name}_{col}" for col in hist.columns]
                hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
                dfs[name] = hist
            except Exception as e:
                print(f"Error occurred while downloading data for {name}: {e}")
                tickers_with_errors.append(ticker)
    elif isinstance(data, list):
        for ticker in data:
            try:
                ticker_obj = yf.Ticker(ticker)
                hist = ticker_obj.history(period=period)
                hist.columns = [f"{ticker}_{col}" for col in hist.columns]
                hist.index = pd.to_datetime(hist.index.map(lambda x: x.strftime('%Y-%m-%d')))
                dfs[ticker] = hist
            except Exception as e:
                print(f"Error occurred while downloading data for {ticker}: {e}")
                tickers_with_errors.append(ticker)
    
    return dfs, tickers_with_errors
