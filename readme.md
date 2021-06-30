# fxSpotRate prediction

Word for task and project documentation: [Intraday FX Spot predictions with state-of-the-art Transformer and Time Embeddings](https://onedrive.live.com/View.aspx?resid=CBDEAB06F9642D15!105)


## Research Ideas:
- [ ] Multivariate FX rate input, e.g. X-USD, X-EUR, X-UKP, X-CHF, ... with reference
- [ ] Adding exogenous time series, e.g. lending rate/volatility/market index
- [ ] Supporting return prediction with text data
    - ['Predicting Returns with Text Data'](http://dx.doi.org/10.2139/ssrn.3389884)

    
## Testing:
x FXTransformer.ipynb: Original file, trained with 1'000 epochs, result: overfitting   MSE=0.06, MAE=0.2, MAPE=1'700

x FXTransformer-Copy1.ipynb: Test file (Scatter Plot für Weekend Data & lineare Regression)

x FXTransformer-Copy2.ipynb: 50 epochs                       MSE=0.004, MAE=0.05, MAPE=300

x FXTransformer-Copy3.ipynb: 100 epochs, Dropout 0.2         MSE=0.004, MAE=0.04, MAPE=200
x FXTransformer-Copy4.ipynb: 200 epochs, Dropout 0.3         MSE=0.01, MAE=0.06, MAPE=400
- FXTransformer-Copy5.ipynb: 500 epochs, Dropout 0.4

x FXTransformer-Copy6.ipynb: 100 epochs, EURCHF              MSE=0.01, MAE=0.04, MAPE=400
x FXTransformer-Copy7.ipynb: 100 epochs, USDCHF25R1M Curncy Trade Open MSE=0.004, MAE=0.04, MAPE=250
x FXTransformer-Copy8.ipynb: 100 epochs, NZDUSD1M Curncy Trade Open MSE=0.004, MAE=0.04, MAPE=250

x FXTransformer-Copy9.ipynb: 100 epochs, Dimensionen reduziert von 256 auf 128 (d_k, d_v & ff_dim = 128 anstatt 256)MSE=0.02, MAE=0.1, MAPE=800

FXTransformer-Copy10.ipynb: 1h
FXTransformer-Copy11.ipynb: 1d
FXTransformer-Copy12.ipynb: 1w



Legende: 
- = wird aktuell trainiert
x = Training abgeschlossen



Paper 1: 15.11.2020 – 15.09.2021

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


Tizian

Marius


