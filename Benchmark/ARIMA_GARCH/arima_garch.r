rm(list=ls())

library(lattice)
library(timeSeries)
library(rugarch)
library(future.apply)
plan(multisession)

setwd('~/github/fxpred/Benchmark/ARIMA_GARCH')

x_test = read.csv('X_test.csv')
x_val = read.csv('X_val.csv')
x_train = read.csv('X_train.csv')

res = expand.grid(
  ar = 1:5,
  ma = 1:5,
  garch_alpha = 1:5,
  garch_beta = 1:5
)
res_info = future_apply(res, MARGIN = 1, function(param) {
  alpha = unlist(unname(param['garch_alpha']))
  beta = unlist(unname(param['garch_beta']))
  p = unlist(unname(param['ar']))
  q = unlist(unname(param['ma']))
  model=ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(alpha, beta)),
    mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
    distribution.model = "norm"
  )
  model_fit = ugarchfit(spec=model, data=x_train)

  model_val = ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(alpha, beta)),
    mean.model = list(armaOrder = c(p, q), include.mean = TRUE),
    distribution.model = "norm",
    fixed.pars = lapply(model_fit@model$pars[, 'Level'], function(x) x)
  )
  model_val_fit = ugarchfit(
    spec=model_val, data=x_val,
    fit.control = list(
      # stationarity = 1, 
      fixed.se = TRUE
      # scale = 0, rec.init = 'all', trunclag = 100
    )
  )
  # model2_fit@model$pars[, 'Level'] - model_fit@model$pars[, 'Level']
  infocriteria(model_val_fit)
})
rownames(res_info) = c("Akaike", "Bayes", "Shibata", "Hannan-Quinn")
res = cbind(res, t(res_info))

write.csv(res, file='arima_garch_param_opt.csv', row.names = FALSE)

