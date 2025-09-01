
# Security Market Line (SML) Analysis App

This Streamlit application allows users to perform Security Market Line (SML) analysis on selected stock tickers. It fetches historical data, calculates risk metrics, and visualizes expected vs. actual returns using the Capital Asset Pricing Model (CAPM).

## Features
- Select default or custom stock tickers
- Choose a date range for analysis
- Fetch historical price data using Yahoo Finance
- Calculate annual returns, arithmetic and geometric means, variance, standard deviation, and coefficient of variation
- Estimate beta and expected return using linear regression
- Visualize the Security Market Line with Plotly
- Compare expected returns with actual returns

## Usage
1. Run the app using Streamlit:
   ```bash
   streamlit run sml_app_multi_tickers.py
   ```
2. Use the sidebar to:
   - Select stock tickers
   - Add custom tickers
   - Set the start and end dates
3. View calculated metrics and interactive plots

## Dependencies
- streamlit
- yfinance
- pandas
- numpy
- plotly
- scikit-learn

Install dependencies using:
```bash
pip install streamlit yfinance pandas numpy plotly scikit-learn
```

## Notes
- Market benchmark used: S&P 500 (`^GSPC`)
- Risk-free rate proxy: 10-Year Treasury Note (`^TNX`)

## License
This project is open-source and available under the MIT License.
