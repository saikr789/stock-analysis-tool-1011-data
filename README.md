## Scraping and Feature Creation

- Run index_scrape file
- - downloads index file
- Run risk_free_rate file
- - downloads the risk free rate
- Run dailystock file
- - downloads bhavcopy
- - downloads deliverables
- - creates a file with stock data all companies for that particular day
- - updates stock files of the respective companies by adding the its respective row from newly downloaded data
- Run feature_creation_daily
- - updates the GRStock files of the respective companies by creating features for newly added rows

## Predict And Simulate

- predict :
- - predict30, predict60 ... predict720
- - Next day Upper and Lower Band for 30,60,90,180,360,720 days
- simulate :
- - newsimulation30, newsimulation60 ... newsimulation720
- - simulation files are created for 30,60,90,180,360,720 days
- - top buy, top sell, suggest files are created with different days

## all the files will be run daily at a particular time by github actions.

# Data Directory Structure

```
|-- stock-analysis-tool-1011-data
   |-- README.md
   |-- corporate_actions_scrape.py
   |-- create_simulation_files.py
   |-- create_top_files.py
   |-- dailystock.py
   |-- feature_creation.py
   |-- feature_creation_daily.py
   |-- index_scrape.py
   |-- newsimulation.py
   |-- newsimulation180.py
   |-- newsimulation30.py
   |-- newsimulation360.py
   |-- newsimulation60.py
   |-- newsimulation720.py
   |-- newsimulation90.py
   |-- predict1080.py
   |-- predict180.py
   |-- predict30.py
   |-- predict360.py
   |-- predict60.py
   |-- predict720.py
   |-- predict90.py
   |-- requirements.txt
   |-- revenue_profit_scrape.py
   |-- risk_free_rate_scrape.py
   |-- simulation.py
   |-- .github
   |   |-- workflows
   |       |-- CorporateActionsScrape.yml
   |       |-- RevenueProfitScrape.yml
   |       |-- performOperationsDailyPredictAndSimulate.yml
   |       |-- performOperationsDailyScrapingAndFeatureCreation.yml
   |-- Data
   |   |-- Equity.csv
   |   |-- Index.csv
   |   |-- RiskFreeRate.csv
   |   |-- RiskFreeRateFull.csv
   |   |-- SP500Companies.json
   |   |-- SP500companies.csv
   |   |-- companies.json
   |   |-- companywithid.json
   |   |-- next_1080_days.csv
   |   |-- next_180_days.csv
   |   |-- next_30_days.csv
   |   |-- next_360_days.csv
   |   |-- next_60_days.csv
   |   |-- next_720_days.csv
   |   |-- next_90_days.csv
   |   |-- sectors.json
   |   |-- sp500.csv
   |   |-- CorporateActions
   |   |   |-- 500002.csv
   |   |   |-- 500003.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- 543210.csv
   |   |   |-- 570001.csv
   |   |-- GRStock
   |   |   |-- gr500002.csv
   |   |   |-- gr500003.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- gr543210.csv
   |   |   |-- gr570001.csv
   |   |-- Revenue
   |   |   |-- 500002.csv
   |   |   |-- 500003.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- 543210.csv
   |   |   |-- 570001.csv
   |   |-- Simulation
   |   |   |-- 500002_1080.csv
   |   |   |-- 500002_180.csv
   |   |   |-- 500002_30.csv
   |   |   |-- 500002_360.csv
   |   |   |-- 500002_60.csv
   |   |   |-- 500002_720.csv
   |   |   |-- 500002_90.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- 570001_1080.csv
   |   |   |-- 570001_180.csv
   |   |   |-- 570001_30.csv
   |   |   |-- 570001_360.csv
   |   |   |-- 570001_60.csv
   |   |   |-- 570001_720.csv
   |   |   |-- 570001_90.csv
   |   |-- SimulationResult
   |   |   |-- 500003_1080.csv
   |   |   |-- 500008_720.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- 570001_1080.csv
   |   |   |-- 570001_720.csv
   |   |   |-- 570001_90.csv
   |   |-- Stock
   |   |   |-- 500002.csv
   |   |   |-- 500003.csv
   |   |   |-- ...
   |   |   |-- ...
   |   |   |-- 543210.csv
   |   |   |-- 570001.csv
   |   |   |-- bhav.csv
   |   |   |-- deliverable.csv
   |   |   |-- previousdaystockdetails.csv
   |   |   |-- result.csv
   |   |-- Top
   |       |-- buy_180.csv
   |       |-- buy_30.csv
   |       |-- buy_360.csv
   |       |-- buy_60.csv
   |       |-- buy_720.csv
   |       |-- buy_90.csv
   |       |-- sell_180.csv
   |       |-- sell_30.csv
   |       |-- sell_360.csv
   |       |-- sell_60.csv
   |       |-- sell_720.csv
   |       |-- sell_90.csv
   |       |-- sim_1080.csv
   |       |-- sim_180.csv
   |       |-- sim_30.csv
   |       |-- sim_360.csv
   |       |-- sim_60.csv
   |       |-- sim_720.csv
   |       |-- sim_90.csv
   |       |-- simres_180.csv
   |       |-- simres_30.csv
   |       |-- simres_360.csv
   |       |-- simres_60.csv
   |       |-- simres_720.csv
   |       |-- simres_90.csv
   |-- ScrapingAndFeatureCreationAndModels
       |-- Models
       |   |-- models.py
       |   |-- regression-lb-ub.ipynb
       |   |-- FinalAlgorithmsIPYNBFiles
       |   |   |-- bpnn-classification.ipynb
       |   |   |-- cnn-classification.ipynb
       |   |   |-- knn-classification.ipynb
       |   |   |-- logistic-classification.ipynb
       |   |   |-- rnn-classification.ipynb
       |   |   |-- svm-classification.ipynb
       |   |-- KaggleFiles
       |       |-- backpropagationneuralnetwork-classification.ipynb
       |       |-- cnn-classification-params.ipynb
       |       |-- cnn-classification.ipynb
       |       |-- cnn-param.ipynb
       |       |-- k-nearest-neighbour-classification.ipynb
       |       |-- logistic-model.ipynb
       |       |-- recurrentneuralnetwork-classification.ipynb
       |       |-- rnn-param.ipynb
       |-- ScrapingAndFeatureCreation
           |-- corporate_actions.py
           |-- corporate_actions_scrape.py
           |-- dailystock.py
           |-- data_cleaning.py
           |-- equity_scrape.py
           |-- feature_creation.py
           |-- feature_creation_daily.py
           |-- index_scrape.py
           |-- revenue_profit_scrape.py
           |-- risk_free_rate_scrape.py
           |-- stock_scrape.py
```
