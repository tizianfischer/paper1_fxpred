# Intraday FX Spot predictions with state-of-the-art Transformer and Time Embeddings



## Documents
- Word for task and project documentation: https://1drv.ms/w/s!AhUtZPkGq97LadLiYdvNgGxE0ys?e=bKDMTo
- Excel for Literature Overview: https://1drv.ms/x/s!AhUtZPkGq97LcjcoWyIsYTdu-Q0?e=UWrirP
- Excel for Experiment Results: https://1drv.ms/x/s!AhUtZPkGq97Lbg4goKL8vqkZ-CQ?e=00qk5d

    

## Explanation of Folders
- Transformer: All Transformer Jupyter Notebooks are saved in this folder
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


6. Daten aufbereiten: Data-Merge, Completeness, Correctness


7. Transformer trainieren, evaluieren & optimieren:
- Overfitting beheben
§ Dropout erhöhen
§ Dimensionen reduzieren
§ Epochs reduzieren
- Verschiedene Kombinationen
§ FX-Pairs: EURUSD, USDJPY, GBPUSD, AUDUSD & USDCAD
§ Alternative Y: Volas (USDCHF25R1M), Swaps (NZDUSD1M), Commodities
§ Verschiedene Time Steps: 1h, 1d
- Anderes
§ Wo Weekend Daten? (Scatter Plot oder X-Achse ohne Datum)
§ Lineare Regression


8. Experimente durchführen: Code von Stefan als Basis


9. Paper schreiben: Word-Skelett als Basis