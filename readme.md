# Intraday FX Spot predictions with state-of-the-art Transformer and Time Embeddings



## Documents
0) Online Folder with Documents: https://1drv.ms/u/s!AhUtZPkGq97LfAgYPV3W4lauTXs?e=acHtOR
1) Paper: https://1drv.ms/w/s!AhUtZPkGq97LekqhMSrrbDL7RZ0?e=hW69mI
2) Task and Project Documentation: https://1drv.ms/w/s!AhUtZPkGq97LdF2JwD0ldXMCm4w?e=bvgMQP
3) Literature Overview: https://1drv.ms/x/s!AhUtZPkGq97LcjcoWyIsYTdu-Q0?e=UWrirP
4) Ongoing Experiments: https://1drv.ms/x/s!AhUtZPkGq97Lbg4goKL8vqkZ-CQ?e=00qk5d
5) Final Experiments Results: https://1drv.ms/x/s!AhUtZPkGq97LgQfV5bwdkFzRtyGU?e=bYlkk7

    

## Explanation of Folders
- Transformer: All Transformer Jupyter Notebooks are saved in this folder
- Benchmark: All Benchmark (ARIMA, RNN & LSTM) Jupyter Notebooks are saved in this folder
- data: Input for Jupyter Notebooks
- content: Output of Jupyter Notebooks



## Overview Paper 1

1. Literaturanalyse: Excel-Übersicht Paper


2. Forschungslücke:
- Transformer with Time Embeddings vs LSTM/RNN/ARIMA
- Multivariate Input Daten vs Univariate Input Daten
- FX
- Intraday


3. Experimente:
A. Systematischer Vergleich Transformer vs State-of-the-art (LSTM, RNN & ARIMA)
B. Systematischer Vergleich Multivariate Input-Daten vs Univariate Input-Daten
- Predicted werden EURUSD, USDJPY, GBPUSD, AUDUSD & USDCAD closing returns
- Evaluationskriterien
§ Statistisch: MSE, MAE, MAPE
§ Ökonomisch: Trading Strategie (Long if Prediction > 0.1% | Short if Prediction < -0.1% | Close after 1 step)
§ Effizienz: Zeit fürs Training


4. Titel: Intraday FX Spot predictions with state-of-the-art Transformer and Time Embeddings


5. Data:
- Multivariat
§ FX Spot Rates
§ FX Volatilitäten, Terminpunkte, RR
§ FX Technical Indicators
§ Fixed Income, Equity, Commodities
- 10 Minuten
- 01.11.2020 – 31.03.2021
- Bloomberg