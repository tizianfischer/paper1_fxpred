rm(list=ls())

library(lattice)
library(timeSeries)
library(rugarch)
library(future.apply)
plan(multisession)
setwd('~/github/fxpred/Benchmark/ARIMA_GARCH/')

# Read data ---------------------------------------------------------------
params = read.csv(file='arima_garch_param_opt.csv')
x_test = read.csv('X_test.csv', header = FALSE)
x_val = read.csv('X_val.csv', header = FALSE)
x_train = read.csv('X_train.csv', header = FALSE)

# Set optimal parameter based on AIC of Validation data -------------------
param = params[order(params$Akaike),][1,]
print(param)
alpha = unlist(unname(param['garch_alpha']))
beta = unlist(unname(param['garch_beta']))
p = unlist(unname(param['ar']))
q = unlist(unname(param['ma']))

# Fit training data model -------------------------------------------------
model=ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(alpha, beta)),
  mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
  distribution.model = "norm"
)
model_fit = ugarchfit(
  spec=model,
  data = rbind(x_train, x_train),
  out.sample = nrow(x_train)
)

# Fit validation data mode ------------------------------------------------
model_val = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(alpha, beta)),
  mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
  distribution.model = "norm",
  fixed.pars = lapply(model_fit@model$pars[, 'Level'], function(x) x)
)
model_val_fit = ugarchfit(
  spec = model_val, 
  data = rbind(x_train, x_val),
  fit.control = list(fixed.se = TRUE),
  out.sample = nrow(x_val)
)

# Fit validation data mode ------------------------------------------------
model_test = ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(alpha, beta)),
  mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
  distribution.model = "norm",
  fixed.pars = lapply(model_fit@model$pars[, 'Level'], function(x) x)
)
model_test_fit = ugarchfit(
  spec = model_test, 
  data = rbind(x_val, x_test), 
  fit.control = list(fixed.se = TRUE),
  out.sample = nrow(x_test)
)

# checking model coefficients ---------------------------------------------
model_test_fit@fit$coef
model_val_fit@fit$coef
model_fit@fit$coef

# Creating and saving forecasts -------------------------------------------
# train
f = ugarchforecast(model_fit, n.ahead = 1, n.roll = nrow(x_train))
x_val_forecast = cbind(t(f@forecast$seriesFor), t(f@forecast$sigmaFor))[1:nrow(x_train), ]
colnames(x_val_forecast) = c('mean', 'sigma')
write.csv(x_val_forecast, file='X_train_forecast.csv', row.names = FALSE)

# validation
f = ugarchforecast(model_val_fit, n.ahead = 1, n.roll = nrow(x_val))
x_val_forecast = cbind(t(f@forecast$seriesFor), t(f@forecast$sigmaFor))[1:nrow(x_val), ]
colnames(x_val_forecast) = c('mean', 'sigma')
write.csv(x_val_forecast, file='X_val_forecast.csv', row.names = FALSE)

# test
f = ugarchforecast(model_test_fit, n.ahead = 1, n.roll = nrow(x_test))
x_test_forecast = cbind(t(f@forecast$seriesFor), t(f@forecast$sigmaFor))[1:nrow(x_test), ]
colnames(x_test_forecast) = c('mean', 'sigma')
write.csv(x_test_forecast, file='X_test_forecast.csv', row.names = FALSE)