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

def download_data(data, period):
    dfs = []

    if isinstance(data, dict):
        # If input is a dictionary, assume keys are names and values are tickers
        for name, ticker in data.items():
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{name}_{col}" for col in hist.columns]  # Add prefix to the name
            dfs.append(hist)
    elif isinstance(data, list):
        # If input is a list, assume tickers directly without names
        for ticker in data:
            ticker_obj = yf.Ticker(ticker)
            hist = ticker_obj.history(period=period)
            hist.columns = [f"{ticker}_{col}" for col in hist.columns]  # Add prefix to the name
            dfs.append(hist)
    else:
        raise ValueError("Input data must be either a dictionary or a list of tickers.")

    df = pd.concat(dfs, axis=1)  # Concatenate along the date index
    return df
    
def load_data_from_github(url):
    response = requests.get(url)
    content = BytesIO(response.content)
    data = pd.read_pickle(content)
    return data

## Configuração da página e do título
st.set_page_config(page_title='Portfolio Balancer', layout = 'wide', initial_sidebar_state = 'auto')
st.title("Portfolio Balancer")

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

currencies_dict  =  {'USD/JPY': 'USDJPY=X', 'USD/BRL': 'BRL=X', 'USD/ARS': 'ARS=X', 'USD/PYG': 'PYG=X', 'USD/UYU': 'UYU=X',
                     'USD/CNY': 'CNY=X', 'USD/KRW': 'KRW=X', 'USD/MXN': 'MXN=X', 'USD/IDR': 'IDR=X', 'USD/EUR': 'EUR=X',
                     'USD/CAD': 'CAD=X', 'USD/GBP': 'GBP=X', 'USD/CHF': 'CHF=X', 'USD/AUD': 'AUD=X', 'USD/NZD': 'NZD=X',
                     'USD/HKD': 'HKD=X', 'USD/SGD': 'SGD=X', 'USD/INR': 'INR=X', 'USD/RUB': 'RUB=X', 'USD/ZAR': 'ZAR=X',
                     'USD/SEK': 'SEK=X', 'USD/NOK': 'NOK=X', 'USD/TRY': 'TRY=X', 'USD/AED': 'AED=X', 'USD/SAR': 'SAR=X',
                     'USD/THB': 'THB=X', 'USD/DKK': 'DKK=X', 'USD/MYR': 'MYR=X', 'USD/PLN': 'PLN=X', 'USD/EGP': 'EGP=X',
                     'USD/CZK': 'CZK=X', 'USD/ILS': 'ILS=X', 'USD/HUF': 'HUF=X', 'USD/PHP': 'PHP=X', 'USD/CLP': 'CLP=X',
                     'USD/COP': 'COP=X', 'USD/PEN': 'PEN=X', 'USD/KWD': 'KWD=X', 'USD/QAR': 'USD/QAR'
                    }
crypto_dict = {'BTC': 'BTC=X', 'ETH': 'ETH=X', 'XRP': 'XRP=X', 'BCH': 'BCH=X', 'LTC': 'LTC=X',
               'EOS': 'EOS=X', 'ADA': 'ADA=X', 'XLM': 'XLM=X', 'LINK': 'LINK=X', 'DOT': 'DOT=X',
               'XMR': 'XMR=X', 'BSV': 'BSV=X', 'TRX': 'TRX=X', 'NEO': 'NEO=X', 'MIOTA': 'MIOTA=X'
              }
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
                    "ALLIAR": "AALR3.SA", "ALPER": "APER3.SA", "ALPHABET": "GOGL35.SA", "ALUPAR INVESTIMENTO": "ALUP4.SA",
                    "AMC ENTERT H": "A2MC34.SA", "AMERICAN EXPRESS": "AXPB34.SA", "APPLE": "AAPL34.SA", "ARCELOR": "ARMT34.SA",
                    "ATT INC": "ATTB34.SA", "AUREN ENERGIA": "AURE3.SA", "AVALARA INC": "A2VL34.SA", "AVON": "AVON34.SA",
                    "BANCO DO BRASIL": "BBAS11.SA", "BANCO INTER": "BIDI3.SA", "BANCO MERCANTIL DE INVESTIMENTOS": "BMIN3.SA",
                    "BANCO PAN": "BPAN4.SA", "BANK AMERICA": "BOAC34.SA", "BANPARA": "BPAR3.SA", "BANRISUL": "BRSR3.SA",
                    "BATTISTELLA": "BTTL3.SA", "BAUMER": "BALM3.SA", "BB SEGURIDADE": "BBSE3.SA", "BEYOND MEAT": "B2YN34.SA",
                    "BIOMM": "BIOM3.SA", "BIOTOSCANA": "GBIO33.SA", "BMG": "BMGB11.SA", "BRASIL BROKERS": "BBRK3.SA",
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
                    "VALE": "VALE5.SA", "VERIZON": "VERZ34.SA", "VISA": "VISA34.SA", "VIVARA": "VIVA3.SA", "VIVEO": "VVEO3.SA", "VOTORANTIM ASSET MANAGEMENT": "VSEC11.SA", "WALMART": "WALM34.SA",
                    "WELLS FARGO": "WFCO34.SA", "WEST PHARMA": "W2ST34.SA", "WILSON SONS": "PORT3.SA", "XEROX": "XRXB34.SA", "XP INC": "XPBR31.SA", "ZYNGA INC": "Z2NG34.SA"
                    }

indexes_dict =     {
                    S&P GSCI': 'GD=F', 'IBOVESPA': '^BVSP', 'S&P/CLX IPSA': '^IPSA',
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
                '3M': 'MMM', 'A. O. SMITH CORPORATION': 'AOS', 'ABBOTT LABORATORIES': 'ABT',
                'ABBVIE INC.': 'ABBV', 'ACCENTURE PLC': 'ACN', 'ACTIVISION BLIZZARD': 'ATVI',
                'ADOBE INC.': 'ADBE', 'ADVANCED MICRO DEVICES INC.': 'AMD', 'AES CORP': 'AES',
                'AFLAC INC.': 'AFL', 'AGILENT TECHNOLOGIES INC': 'A', 'AIR PRODUCTS AND CHEMICALS INC': 'APD',
                'AKAMAI TECHNOLOGIES INC': 'AKAM', 'ALASKA AIR GROUP INC': 'ALK', 'ALBEMARLE CORPORATION': 'ALB',
                'ALEXANDRIA REAL ESTATE EQUITIES': 'ARE', 'ALIGN TECHNOLOGY': 'ALGN', 'ALLEGIANT': 'ALLE',
                'ALLIANT ENERGY CORP': 'LNT', 'ALLSTATE CORP': 'ALL', 'ALPHABET INC. (CLASS A)': 'GOOGL',
                'ALPHABET INC. (CLASS C)': 'GOOG', 'ALTRIA GROUP INC': 'MO', 'AMAZON.COM INC.': 'AMZN',
                'AMCOR PLC': 'AMCR', 'AMEREN CORP': 'AEE', 'AMERICAN AIRLINES GROUP': 'AAL',
                'AMERICAN ELECTRIC POWER': 'AEP', 'AMERICAN EXPRESS': 'AXP', 'AMERICAN INTERNATIONAL GROUP': 'AIG',
                'AMERICAN TOWER CORP.': 'AMT', 'AMERICAN WATER WORKS COMPANY INC': 'AWK', 'AMERIPRISE FINANCIAL': 'AMP',
                'AMERISOURCEBERGEN CORP': 'ABC', 'AMETEK': 'AME', 'AMGEN INC.': 'AMGN', 'AMPHENOL CORP': 'APH',
                'ANSYS': 'ANSS', 'AON PLC': 'AON', 'A. O. SMITH CORPORATION': 'AOS', 'APA CORPORATION': 'APA',
                'APPLE INC.': 'AAPL', 'APPLIED MATERIALS INC.': 'AMAT', 'ARCH CAPITAL GROUP LTD.': 'ACGL',
                'ARISTA NETWORKS': 'ANET', 'ARROW ELECTRONICS INC.': 'ARW', 'ASSURANT INC.': 'AIZ',
                'AT&T INC.': 'T', 'AUTODESK INC.': 'ADSK', 'AUTOZONE INC.': 'AZO', 'AVALONBAY COMMUNITIES INC.': 'AVB',
                'BAKER HUGHES CO.': 'BKR', 'BALL CORPORATION': 'BALL', 'BANK OF AMERICA CORP.': 'BAC',
                'BAXTER INTERNATIONAL INC.': 'BAX', 'BECTON DICKINSON AND CO.': 'BDX', 'BERKSHIRE HATHAWAY INC.': 'BRK.B',
                'BEST BUY CO. INC.': 'BBY', 'BIO-RAD LABORATORIES INC.': 'BIO', 'BIO-TECHNE CORPORATION': 'TECH',
                'BLACKROCK INC.': 'BLK', 'BOEING COMPANY': 'BA', 'BOOKING HOLDINGS INC.': 'BKNG',
                'BOSTON PROPERTIES INC.': 'BXP', 'BOSTON SCIENTIFIC CORPORATION': 'BSX', 'BRISTOL-MYERS SQUIBB CO.': 'BMY',
                'BROADCOM INC.': 'AVGO', 'BROADRIDGE FINANCIAL SOLUTIONS INC.': 'BR', 'BRUNSWICK CORPORATION': 'BC',
                'BURLINGTON STORES INC.': 'BURL', 'CADENCE DESIGN SYSTEMS INC.': 'CDNS', 'CAESARS ENTERTAINMENT INC.': 'CZR',
                'CAIGNET INC.': 'CZR', 'CAMPBELL SOUP COMPANY': 'CPB', 'CAMPING WORLD HOLDINGS INC.': 'CWH',
                'CARDINAL HEALTH INC.': 'CAH', 'CARMAX INC.': 'KMX', 'CARNIVAL CORPORATION & PLC': 'CCL',
                'CARVANA CO.': 'CVNA', 'CATERPILLAR INC.': 'CAT', 'CBOE GLOBAL MARKETS INC.': 'CBOE',
                'CBRE GROUP INC.': 'CBRE', 'CDW CORPORATION': 'CDW', 'CELANESE CORPORATION': 'CE',
                'CENTENE CORPORATION': 'CNC', 'CENTERPOINT ENERGY INC.': 'CNP', 'CERNER CORPORATION': 'CERN',
                'CF INDUSTRIES HOLDINGS INC.': 'CF', 'CHARLES RIVER LABORATORIES INTERNATIONAL INC.': 'CRL',
                'CHARLES SCHWAB CORPORATION': 'SCHW', 'CHARTER COMMUNICATIONS INC.': 'CHTR', 'CHEGG INC.': 'CHGG',
                'CHEMED CORPORATION': 'CHE', 'CHEMOURS COMPANY': 'CC', 'CHEROKEE INC.': 'CHKE',
                'CHESAPEAKE ENERGY CORPORATION': 'CHK', 'CHEVRON CORPORATION': 'CVX', 'CHIPOTLE MEXICAN GRILL INC.': 'CMG',
                'CHUBB LIMITED': 'CB', 'CHURCH & DWIGHT CO. INC.': 'CHD', 'CIGNA CORPORATION': 'CI',
                'CIMAREX ENERGY CO.': 'XEC', 'CINCINNATI FINANCIAL CORPORATION': 'CINF', 'CINTAS CORPORATION': 'CTAS',
                'CISCO SYSTEMS INC.': 'CSCO', 'CITIGROUP INC.': 'C', 'CITIZENS FINANCIAL GROUP INC.': 'CFG',
                'CLOROX COMPANY': 'CLX', 'CME GROUP INC.': 'CME', 'CMS ENERGY CORPORATION': 'CMS',
                'CNA FINANCIAL CORPORATION': 'CNA', 'CNH INDUSTRIAL NV': 'CNHI', 'COCA-COLA COMPANY': 'KO',
                'COGNIZANT TECHNOLOGY SOLUTIONS CORPORATION': 'CTSH', 'COHR INC.': 'COHR',
                'COINSTAR INC.': 'CSTR', 'COLDWELL BANKER REALTY': 'NRT', 'COLDWELL BANKER REALTY': 'NRT', 'COLFAX CORPORATION': 'CFX',
                'COLUMBIA PROPERTY TRUST INC.': 'CXP','COLUMBUS MCKINNON CORPORATION': 'CMCO', 'COMCAST CORPORATION': 'CMCSA', 'COMERICA INC.': 'CMA',
                'COMFORT SYSTEMS USA INC.': 'FIX', 'COMPASS MINERALS INTERNATIONAL INC.': 'CMP', 'CONAGRA BRANDS INC.': 'CAG',
                'CONCHO RESOURCES INC.': 'CXO', 'CONDUENT INC.': 'CNDT', 'CONOCOPHILLIPS': 'COP', 'CONSOLIDATED EDISON INC.': 'ED',
                'CONSTELLATION BRANDS INC.': 'STZ', 'COOPER COMPANIES INC.': 'COO', 'COOPER STANDARD HOLDINGS INC.': 'CPS',
                'COPART INC.': 'CPRT', 'CORNING INC.': 'GLW', 'CORTYNE WORKFLOW (ACTIVATED BY INDIVIDUAL)': 'CNNE',
                'COSTAR GROUP INC.': 'CSGP', 'COSTCO WHOLESALE CORPORATION': 'COST', 'COTY INC.': 'COTY',
                'COURIER CORPORATION': 'CDW', 'COWEN INC.': 'COWN', 'CRANE COMPANY': 'CR', 'CREDIT ACCEPTANCE CORPORATION': 'CACC',
                'CREE INC.': 'CREE', 'CROWN CASTLE INTERNATIONAL CORP.': 'CCI', 'CROWN HOLDINGS INC.': 'CCK',
                'CSX CORPORATION': 'CSX', 'CULLEN/FROST BANKERS INC.': 'CFR', 'CUMMINS INC.': 'CMI', 'CURTISS-WRIGHT CORPORATION': 'CW',
                'CVS HEALTH CORPORATION': 'CVS', 'DAIMLER AG': 'DDAIF', 'DANAHER CORPORATION': 'DHR', 'DARDEN RESTAURANTS INC.': 'DRI',
                'DARLING INGREDIENTS INC.': 'DAR', 'DAVITA INC.': 'DVA', 'DECKERS OUTDOOR CORPORATION': 'DECK',
                'DEERE & COMPANY': 'DE', 'DELTA AIR LINES INC.': 'DAL', 'DELL TECHNOLOGIES INC.': 'DELL',
                'DENBURY INC.': 'DEN', 'DEXCOM INC.': 'DXCM', 'DIAGEO PLC': 'DEO', 'DIAMOND BACK ENERGY INC.': 'FANG',
                'DIGITAL REALTY TRUST INC.': 'DLR', 'DISCO CORPORATION': 'DSCO', 'DISCOVER FINANCIAL SERVICES': 'DFS',
                'DISCOVERY INC.': 'DISCA', 'DISH NETWORK CORPORATION': 'DISH', 'DOLLAR GENERAL CORPORATION': 'DG',
                'DOLLAR TREE INC.': 'DLTR', 'DOMINION ENERGY INC.': 'D', 'DONALDSON COMPANY INC.': 'DCI',
                'DOVER CORPORATION': 'DOV', 'DOW INC.': 'DOW', 'DRAFTKINGS INC.': 'DKNG', 'DTE ENERGY COMPANY': 'DTE',
                'DUKE ENERGY CORPORATION': 'DUK', 'DUKE REALTY CORPORATION': 'DRE', 'DUN & BRADSTREET HOLDINGS INC.': 'DNB',
                'DXC TECHNOLOGY COMPANY': 'DXC', 'DYCOM INDUSTRIES INC.': 'DY', 'DYNATRACE INC.': 'DT',
                'E*TRADE FINANCIAL CORPORATION': 'ETFC', 'EAST WEST BANCORP INC.': 'EWBC', 'EASTMAN CHEMICAL COMPANY': 'EMN',
                'EATON CORPORATION PLC': 'ETN', 'EBAY INC.': 'EBAY', 'EHEALTH INC.': 'EHTH', 'ELANCO ANIMAL HEALTH INC.': 'ELAN',
                'ELECTRONIC ARTS INC.': 'EA', 'EMERSON ELECTRIC COMPANY': 'EMR', 'ENCOMPASS HEALTH CORPORATION': 'EHC',
                'ENDEAVOUR SILVER CORP.': 'EXK', 'ENERSYS': 'ENS', 'ENERGY TRANSFER LP': 'ET', 'ENPHASE ENERGY INC.': 'ENPH',
                'ENTEGRIS INC.': 'ENTG', 'ENVESTNET INC.': 'ENV', 'EOG RESOURCES INC.': 'EOG', 'EPAM SYSTEMS INC.': 'EPAM',
                'EQUINIX INC.': 'EQIX', 'EQUITY RESIDENTIAL': 'EQR', 'ESSEX PROPERTY TRUST INC.': 'ESS',
                'ESTEE LAUDER COMPANIES INC.': 'EL', 'EURONET WORLDWIDE INC.': 'EEFT', 'EVERGY INC.': 'EVRG',
                'EVERSOURCE ENERGY': 'ES', 'EVERTEC INC.': 'EVTC', 'EXELON CORPORATION': 'EXC', 'EXPEDITORS INTERNATIONAL OF WASHINGTON INC.': 'EXPD',
                'EXPEDIA GROUP INC.': 'EXPE', 'EXPEDITORS INTERNATIONAL OF WASHINGTON INC.': 'EXPD', 'EXTRA SPACE STORAGE INC.': 'EXR',
                'EXXON MOBIL CORPORATION': 'XOM', 'F5 NETWORKS INC.': 'FFIV', 'FACEBOOK INC.': 'FB', 'FASTENAL COMPANY': 'FAST',
                'FEDEX CORPORATION': 'FDX', 'FERRARI N.V.': 'RACE', 'FIDELITY NATIONAL INFORMATION SERVICES INC.': 'FIS',
                'FIDELITY NATIONAL FINANCIAL INC.': 'FNF', 'FIFTH THIRD BANCORP': 'FITB', 'FIRST AMERICAN FINANCIAL CORPORATION': 'FAF',
                'FIRST DATA CORPORATION': 'FDC', 'FIRST HORIZON CORPORATION': 'FHN', 'FIRST INDUSTRIAL REALTY TRUST INC.': 'FR',
                'FIRST REPUBLIC BANK': 'FRC', 'FIRST SOLAR INC.': 'FSLR', 'FIRSTENERGY CORP.': 'FE',
                'FISERV INC.': 'FISV', 'FIVE BELOW INC.': 'FIVE', 'FLEETCOR TECHNOLOGIES INC.': 'FLT',
                'FLEETCOR TECHNOLOGIES INC.': 'FLT', 'FLIR SYSTEMS INC.': 'FLIR', 'FLOWERS FOODS INC.': 'FLO',
                'FLUOR CORPORATION': 'FLR', 'FLYING EAGLE ACQUISITION CORP.': 'FEAC', 'FOOT LOCKER INC.': 'FL',
                'FORD MOTOR COMPANY': 'F', 'FORTESCUE METALS GROUP LTD.': 'FSUGY', 'FORTINET INC.': 'FTNT',
                'FORTIVE CORPORATION': 'FTV', 'FORTUNE BRANDS HOME & SECURITY INC.': 'FBHS', 'FOX CORPORATION': 'FOX',
                'FRANKLIN RESOURCES INC.': 'BEN', 'FREEPORT-MCMORAN INC.': 'FCX', 'FRESHPET INC.': 'FRPT',
                'FRONTLINE LTD.': 'FRO', 'GALAXY DIGITAL HOLDINGS LTD.': 'BRPHF', 'GALLAGHER ARTHUR J. & CO.': 'AJG',
                'GAP INC.': 'GPS', 'GARMIN LTD.': 'GRMN', 'GARTNER INC.': 'IT', 'GENERAL DYNAMICS CORPORATION': 'GD',
                'GENERAL ELECTRIC COMPANY': 'GE', 'GENERAL ELECTRIC COMPANY': 'GE', 'GENERAL MILLS INC.': 'GIS',
                'GENERAL MOTORS COMPANY': 'GM', 'GENESCO INC.': 'GCO', 'GENESEE & WYOMING INC.': 'GWR',
                'GENESIS HEALTHCARE INC.': 'GEN', 'GENTEX CORPORATION': 'GNTX', 'GENTHERM INC.': 'THRM',
                'GENUINE PARTS COMPANY': 'GPC', 'GENWORTH FINANCIAL INC.': 'GNW', 'GEO GROUP INC.': 'GEO',
                'GERDAU S.A.': 'GGB', 'GIGEASY CORP.': 'GIGE', 'GILDAN ACTIVEWEAR INC.': 'GIL',
                'GILEAD SCIENCES INC.': 'GILD', 'GLAUKOS CORPORATION': 'GKOS', 'GLOBAL PAYMENTS INC.': 'GPN',
                'GLOBE LIFE INC.': 'GL', 'GLOBE TELECOM INC.': 'GTMEF', 'GLU MOBILE INC.': 'GLUU',
                'GMS INC.': 'GMS', 'GODADDY INC.': 'GDDY', 'GOLD RESOURCE CORPORATION': 'GORO',
                'GOLDMAN SACHS GROUP INC.': 'GS', 'GOLDMAN SACHS GROUP INC.': 'GS', 'GOODYEAR TIRE & RUBBER COMPANY': 'GT',
                'GOPRO INC.': 'GPRO', 'GRAINGER W.W. INC.': 'GWW', 'GRAND CANYON EDUCATION INC.': 'LOPE',
                'GRAPHIC PACKAGING HOLDING COMPANY': 'GPK', 'GRAY TELEVISION INC.': 'GTN', 'GREAT PANTHER MINING LIMITED': 'GPL',
                'GREEN PLAINS INC.': 'GPRE', 'GREENE KING PLC': 'GKNGY', 'GREENLIGHT CAPITAL RE LTD.': 'GLRE',
                'GRIFOLS S.A.': 'GRFS', 'GROUPON INC.': 'GRPN', 'GROWGENERATION CORP.': 'GRWG',
                'GRUPO AEROPORTUARIO DEL PACIFICO S.A.B. DE C.V.': 'PAC', 'GRUPO AEROPORTUARIO DEL SURESTE S.A.B. DE C.V.': 'ASR',
                'GRUPO BIMBO S.A.B. DE C.V.': 'BIMBOA', 'GRUPO BIMBO S.A.B. DE C.V.': 'BIMBOA', 'GRUPO FINANCIERO BANORTE S.A.B. DE C.V.': 'GBOOY',
                'GRUPO MEXICO S.A.B. DE C.V.': 'GMBXF', 'GRUPO TELEVISA S.A.B.': 'TV', 'GUARDANT HEALTH INC.': 'GH',
                'GUARDAWORLD CORP.': 'GW', 'GUARDIAN HEALTH SCIENCES LTD.': 'GHSI', 'GUESS? INC.': 'GES',
                'GUIDEWIRE SOFTWARE INC.': 'GWRE', 'GULFPORT ENERGY CORPORATION': 'GPOR', 'H.B. FULLER COMPANY': 'FUL',
                'H&R BLOCK INC.': 'HRB', 'H&E EQUIPMENT SERVICES INC.': 'HEES', 'HALLIBURTON COMPANY': 'HAL',
                'HALOZYME THERAPEUTICS INC.': 'HALO', 'HANCOCK WHITNEY CORPORATION': 'HWC', 'HANESBRANDS INC.': 'HBI',
                'HANNON ARMSTRONG SUSTAINABLE INFRASTRUCTURE CAPITAL INC.': 'HASI', 'HANOVER INSURANCE GROUP INC.': 'THG',
                'HARBORONE BANCORP INC.': 'HONE', 'HARLEY-DAVIDSON INC.': 'HOG', 'HARMONIC INC.': 'HLIT',
                'HARTFORD FINANCIAL SERVICES GROUP INC.': 'HIG', 'HASBRO INC.': 'HAS', 'HAWAIIAN HOLDINGS INC.': 'HA',
                'HCA HEALTHCARE INC.': 'HCA', 'HCC INSURANCE HOLDINGS INC.': 'HCC', 'HCP INC.': 'PEAK',
                'HDFC BANK LIMITED': 'HDB', 'HEALTHCARE REALTY TRUST INC.': 'HR', 'HEALTHCARE SERVICES GROUP INC.':
                 'HEALTHCARE SERVICES GROUP INC.': 'HCSG', 'HEALTHPEAK PROPERTIES INC.': 'PEAK',
                'HEALTHCARE REALTY TRUST INC.': 'HR', 'HEALTHCARE TRUST OF AMERICA INC.': 'HTA',
                'HEARTLAND FINANCIAL USA INC.': 'HTLF', 'HEAT BIOLOGICS INC.': 'HTBX',
                'HECLA MINING COMPANY': 'HL', 'HEICO CORPORATION': 'HEI', 'HELEN OF TROY LIMITED': 'HELE',
                'HELMERICH & PAYNE INC.': 'HP', 'HERBALIFE NUTRITION LTD.': 'HLF',
                'HERC HOLDINGS INC.': 'HRI', 'HERITAGE INSURANCE HOLDINGS INC.': 'HRTG',
                'HERON THERAPEUTICS INC.': 'HRTX', 'HESS CORPORATION': 'HES',
                'HEWLETT PACKARD ENTERPRISE COMPANY': 'HPE', 'HFF INC.': 'HF', 'HIBBETT SPORTS INC.': 'HIBB',
                'HILTON WORLDWIDE HOLDINGS INC.': 'HLT', 'HNI CORPORATION': 'HNI',
                'HOLOGIC INC.': 'HOLX', 'HOME BANCSHARES INC.': 'HOMB', 'HOME DEPOT INC.': 'HD',
                'HOME STREET INC.': 'HMST', 'HOMEAWAY INC.': 'AWAY', 'HONEYWELL INTERNATIONAL INC.': 'HON',
                'HOOKER FURNITURE CORPORATION': 'HOFT', 'HORIZON BANCORP INC.': 'HBNC',
                'HORMEL FOODS CORPORATION': 'HRL', 'HOULIHAN LOKEY INC.': 'HLI',
                'HOUSE COM STK': 'HOUSE', 'HOUSTON AMERICAN ENERGY CORPORATION': 'HUSA',
                'HOWARD HUGHES CORPORATION': 'HHC', 'HP INC.': 'HPQ', 'HSBC HOLDINGS PLC': 'HSBC',
                'HUDSON PACIFIC PROPERTIES INC.': 'HPP', 'HUMANA INC.': 'HUM',
                'HUNT J B TRANS SVCS INC.': 'JBHT', 'HUNTINGTON BANCSHARES INC.': 'HBAN',
                'HUNTSMAN CORPORATION': 'HUN', 'HURON CONSULTING GROUP INC.': 'HURN',
                'HYATT HOTELS CORPORATION': 'H', 'IAC/INTERACTIVECORP': 'IAC',
                'IBERIABANK CORP.': 'IBKC', 'IBERIABANK CORPORATION': 'IBKC', 'ICF INTERNATIONAL INC.': 'ICFI',
                'ICICI BANK LIMITED': 'IBN', 'ICON PLC': 'ICLR', 'IDACORP INC.': 'IDA',
                'IDEAL POWER INC.': 'IPWR', 'IDEX CORPORATION': 'IEX', 'IFF INC.': 'IFF',
                'II-VI INC.': 'IIVI', 'ILLINOIS TOOL WORKS INC.': 'ITW', 'ILLUMINA INC.': 'ILMN',
                'IMAX CORPORATION': 'IMAX', 'IMMERSION CORPORATION': 'IMMR', 'IMPERIAL OIL LIMITED': 'IMO',
                'INCYTE CORPORATION': 'INCY', 'INDEPENDENCE HOLDING COMPANY': 'IHC',
                'INDEPENDENT BANK CORP.': 'INDB', 'INDEPENDENT BANK GROUP INC.': 'IBTX',
                'INDUSTRIAL LOGISTICS PROPERTIES TRUST': 'ILPT', 'INFINERA CORPORATION': 'INFN',
                'INFINITY PHARMACEUTICALS INC.': 'INFI', 'INFORMATION SERVICES GROUP INC.': 'III',
                'INGERSOLL RAND INC.': 'IR', 'INPHI CORPORATION': 'IPHI', 'INSIGHT ENTERPRISES INC.': 'NSIT',
                'INSIGHT SELECT INCOME FUND': 'INSI', 'INSMED INC.': 'INSM', 'INSPERITY INC.': 'NSP',
                'INSPERITY INC.': 'NSP', 'INSULET CORPORATION': 'PODD', 'INTEGRA LIFESCIENCES HOLDINGS CORP.': 'IART',
                'INTEL CORPORATION': 'INTC', 'INTELLIA THERAPEUTICS INC.': 'NTLA', 'INTERCEPT PHARMACEUTICALS INC.': 'ICPT',
                'INTERCONTINENTAL EXCHANGE INC.': 'ICE', 'INTERFACE INC.': 'TILE',
                'INTERMEDIATE CAPITAL GROUP PLC': 'ICP', 'INTERNATIONAL BUSINESS MACHINES CORPORATION': 'IBM',
                'INTERNATIONAL FLAVORS & FRAGRANCES INC.': 'IFF', 'INTERNATIONAL GAME TECHNOLOGY PLC': 'IGT',
                'INTERNATIONAL MONEY EXPRESS INC.': 'IMXI', 'INTERNATIONAL PAPER COMPANY': 'IP',
                'INTERNATIONAL SEAWAYS INC.': 'INSW', 'INTERNATIONAL SPEEDWAY CORPORATION': 'ISCA',
                'INTERNATIONAL TELECOMMUNICATION UNION': 'ITU', 'INTERNET INCOME SOURCE': 'IIS',
                'INTERPUBLIC GROUP OF COMPANIES INC.': 'IPG', 'INTUIT INC.': 'INTU', 'INTUITIVE SURGICAL INC.': 'ISRG'}

assets_dict =  {**currencies_dict, **crypto_dict, **stocks_dict, **sp500_dict, **indexes_dict}

selected_dict_key = st.selectbox('Select the asset type', list(assets_dict.keys()))
if selected_dict_key:
    selected_dict = assets_dict[selected_dict_key]
    selected_assets = st.multiselect(f'Select {selected_dict_key} assets', list(selected_dict.keys()))
    selected_assets_info = {key: selected_dict[key] for key in selected_assets}
    tickers_list = st.multiselect('Available Tickers:', list(selected_assets_info.keys()))

type_tickers = st.text_input('Enter Tickers (comma-separated):')
if type_tickers:    
    tickers = [ticker.strip() for ticker in type_tickers.split(',')]
selected_timeframe = st.selectbox('Select Timeframe:', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])

if st.button("Download data"):
    session_state.data = donwload_data(tickers, selected_timeframe)
    if session_state.data is not None:
        st.dataframe(session_state.data)



