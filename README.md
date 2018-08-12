# Autodidactic trading robot

In this work we want to see whether machine learning algorithms can be used to automatically learn given strategy from decisions made by a trader without requiring any additional input other than sell/buy signal provided by trader and stock market data. We experiment with three methods which can be used to learn to imitate trader's signal. We arrived at a conclusion that while these methods works for simple trading strategies, both are not generalizable enough. Lastly, we attempt to explain why this is a hard problem to solve using a single, generic machine learning algorithm

## Code structure

This repository has following layout:

* data - data, captured from Yahoo Finance
* notebooks - various notebooks, including strategies, models, exploratory analysis
* report, presentation - presentation materials
* src - common code, shared across notebooks

## How to run this code?

To run strategies and scrape data you will need R programming environment. R code can be found in _notebooks/R_ folder of the repository. Most of the models are written in Python and require Jupyter notebook, [Deap](https://github.com/DEAP/deap) and [Keras](https://keras.io/) Python packages
