
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.title("Security Market Line (SML) Analysis")

st.sidebar.header("Configuration")
tickers = st.sidebar.multiselect("Select Tickers", ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], default=['AAPL', 'MSFT', 'GOOGL'])
market_ticker = '^GSPC'
risk_free_ticker = '^TNX'
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2025-01-01'))

try:
    data = yf.download(tickers + [market_ticker], start=start_date, end=end_date)['Close']
    returns = data.pct_change().dropna()
    annual_returns = returns.resample('Y').apply(lambda x: (x + 1).prod() - 1)
    if annual_returns.empty:
        st.error("No data available for the selected date range.")
    else:
        hpr = annual_returns + 1
        arithmetic_mean = annual_returns.mean()
        geometric_mean = hpr.prod()**(1/len(hpr)) - 1
        variance = annual_returns.var()
        std_dev = annual_returns.std()
        coeff_var = std_dev / arithmetic_mean

        st.subheader("Risk Metrics")
        st.write("Arithmetic Mean:", arithmetic_mean)
        st.write("Geometric Mean:", geometric_mean)
        st.write("Variance:", variance)
        st.write("Standard Deviation:", std_dev)
        st.write("Coefficient of Variation:", coeff_var)

        risk_free_data = yf.download(risk_free_ticker, start=start_date, end=end_date)['Close']
        average_risk_free_rate = risk_free_data.mean() / 100
        rf_rate_scalar = float(average_risk_free_rate)

        X = annual_returns[[market_ticker]]
        expected_returns = {}
        betas = {}

        for ticker in tickers:
            y = annual_returns[[ticker]]
            model = LinearRegression().fit(X, y)
            beta = model.coef_[0][0]
            expected_return = rf_rate_scalar + beta * (X.mean().values[0] - rf_rate_scalar)
            expected_returns[ticker] = expected_return
            betas[ticker] = beta

        sml_df = pd.DataFrame({
            'Ticker': tickers,
            'Beta': [betas[t] for t in tickers],
            'Expected Return': [expected_returns[t] for t in tickers]
        })

        risk_free_row = pd.DataFrame({
            'Ticker': ['Risk-Free'],
            'Beta': [0.0],
            'Expected Return': [rf_rate_scalar]
        })

        sml_df = pd.concat([sml_df, risk_free_row], ignore_index=True)

        fig = px.scatter(sml_df, x='Beta', y='Expected Return', text='Ticker',
                         title='Security Market Line (SML)',
                         labels={'Beta': 'Beta', 'Expected Return': 'Expected Return'},
                         template='plotly_white')

        min_beta = sml_df['Beta'].min()
        max_beta = sml_df['Beta'].max()
        sml_x = np.array([min_beta, max_beta])
        sml_y = rf_rate_scalar + sml_x * (X.mean().values[0] - rf_rate_scalar)

        fig.add_trace(go.Scattergl(x=sml_x, y=sml_y, mode='lines', name='SML',
                                   line=dict(color='red', width=2)))
        fig.update_traces(marker=dict(size=12), textposition='top center')

        st.plotly_chart(fig)
except Exception as e:
    st.error(f"An error occurred: {e}")
