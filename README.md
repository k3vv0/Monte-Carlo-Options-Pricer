# Monte Carlo Options Pricing

This repository contains Python scripts for pricing options contracts using Monte Carlo simulations. The primary script is `MCOP.py` which provides the user an interface to select options contracts from a specified ticker. Then it uses Monte Carlo simulations to estimate the value of the selected contract.

## Installation

This project uses a variety of Python libraries which can be installed using the provided `requirements.txt` file:
```sh
pip install -r requirements.txt
```
## Usage

You can run the script in a Python environment with:
```sh
python MCOP.py
```

In the sample script, you will be asked to input a stock ticker symbol. The program will then fetch option contract data for that ticker from Yahoo Finance and present you with a table of contracts within a given date range. You can navigate through the list using the left and right arrow keys, and select a contract by inputting the corresponding row number.

The program then uses a Monte Carlo simulation to estimate the value of the selected option contract and prints the result.

Please note, when running this script, it must have privileges to listen for global keypress events.

## Algorithms & Math

Monte Carlo simulations work by simulating a large number of possible paths for the price of the underlying asset, each of which corresponds to a specific payoff of the option. The average of these payoffs is then calculated, and discounted back to today using the risk-free rate. This average discounted payoff is the estimated value of the option.

The key concept here is that the future price of the underlying asset is modeled as a random process, specifically a geometric Brownian motion, which assumes that the price changes are normally distributed.

For each simulated path, the code generates a random number to represent the change in the asset's price over the option's life. This number is drawn from a normal distribution with mean equal to `(r - 0.5 * vol^2) * T` and standard deviation equal to `vol * sqrt(T)`, where `r` is the risk-free rate, `vol` is the volatility of the underlying asset, and `T` is the time to the option's expiry.

## Upcoming Features

This code is actively being developed and there are several exciting features planned for future updates, including:

- Enabling the user to find contracts by a specified range of strike prices.
- Enabling the user to find contracts by a specified range of ask prices.
- Incorporating actual current risk-free rates into the simulation.
- Risk measurement features to evaluate accuracy of simulation

---
