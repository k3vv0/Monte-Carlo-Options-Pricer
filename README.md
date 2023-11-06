# Monte Carlo Options Pricing

## Disclaimer

The code and methods included in this repository are for educational purposes only. The option pricing estimates provided by the scripts should not be used as the sole basis for making real trading decisions. While the algorithms implemented can provide insights into the theoretical pricing of options contracts, they do not account for all market conditions, behavioral factors, or events that can influence the actual market price of an option. Users are strongly advised to conduct comprehensive market analysis and consult financial advisors before engaging in options trading.

---

## Introduction
This repository contains Python scripts for pricing options contracts using Monte Carlo simulations. The primary script is `MCOP.py` which provides the user an interface to select options contracts from a specified ticker. Then it uses Monte Carlo simulations to estimate the value of the selected contract.

## Prerequisites
Before running the MCOP, ensure that you have the following prerequisites installed:

- Python 3.x
- Git (optional, for cloning the repository)

## Installation

This project uses a variety of Python libraries which can be installed using the provided `requirements.txt` file:
```sh
pip install -r requirements.txt
```
## Setting Up Environment
MCOP uses the FRED API to get the latest risk-free rate data. To set this up:

- Obtain a free API key from the FRED website by registering at https://fred.stlouisfed.org/docs/api/api_key.html.
- Create a .env file in the root directory of the MCOP project.
- Add your FRED API key to the .env file with the variable name FRED_API_KEY:
```
FRED_API_KEY=your_api_key_here
```

This setup allows MCOP to securely access the risk-free rate data.

---

## Usage
This section is crafted to guide a new user step by step through the process of running the script and understanding the output. It includes the default behavior of the script and notes on how to adjust settings for different scenarios.

The Monte Carlo Options Pricer (MCOP) is a Python script that estimates the fair price of options contracts using three Monte Carlo simulation techniques. By default, the script allows the user to select an option contract by date and then computes the price using all three pricers. Follow the steps below to run the script and calculate the prices:

1. **Run the Script**

   Navigate to the directory containing `MCOP.py` and run the script using Python:

   ```bash
   python MCOP.py
   ```
3. **Enter Security Ticker**

   When prompted, enter the ticker symbol for the security you're interested in. For example, `AAPL` for Apple Inc.

4. **Select Contract Date**

   Choose from a list of available option contract expiry dates for the entered ticker. Enter the number corresponding to your chosen date.

5. **Select a Contract**

   After selecting a date, you'll be presented with a list of available contracts for that date. Enter the number corresponding to the specific contract you want to price. By default, the program selects call      options. If you wish to analyze put options, you will need to set `call=False` when instantiating an instance of the `Option` class within the code.

6. **Review Results**

   The script will automatically calculate the estimated fair price for the selected contract using all three pricing classes: `SimpleMCPricer`, `AntitheticMCPricer`, and `LeastSquaresMonteCarloPricer`. The results,     including the fair price estimate, confidence interval, and standard error, will be printed to the console for each pricer.

For a more customized usage or to analyze a different type of options (e.g., put options), you may modify the parameters in the script accordingly. Ensure you have the necessary Python packages installed, as listed in `requirements.txt`.

Remember, the effectiveness of the pricing estimates can be influenced by the volatility of the market, the choice of the risk-free rate, and the specific parameters set for the simulation. Always review the assumptions and parameters to ensure they are appropriate for the contracts being evaluated.

Please note, when running this script, it must have privileges to listen for global keypress events.

## Algorithms & Math

The MCOP script uses three distinct Monte Carlo simulation-based methods to estimate the fair price of options contracts. These methods leverage the power of random sampling to model the uncertainty and dynamics of option prices. Below is an overview of the algorithms and the mathematical concepts underlying them:

### Simple Monte Carlo Pricer (`SimpleMCPricer`)

This method estimates the option price by simulating the path of the underlying security's price over time using a geometric Brownian motion model. The key components include:

- **Spot Price**: The current trading price of the underlying security.
- **Volatility (σ)**: The annualized standard deviation of the security's returns, which measures the price's variability.
- **Risk-Free Rate (r)**: The theoretically risk-free return rate over time, often derived from government securities like Treasury bills.
- **Expiry (T)**: The time until the option's expiration.

The algorithm generates random paths for the underlying security's price using the following stochastic differential equation:

\[
dS = rSdt + \sigma Sdz
\]

where \( S \) is the security price, \( dt \) is a small time step, and \( dz \) represents the random component generated from a standard normal distribution.

The payoff for each path at expiry is calculated, and the mean of these payoffs is discounted back to the present value using the risk-free rate. This mean represents the estimated fair price of the option.

### Antithetic Variates Method (`AntitheticMCPricer`)

The antithetic variates method enhances the simple Monte Carlo approach by reducing variance in the simulation. It does this by generating pairs of random variables where one is the negative of the other (antithetic variates). This leads to a more stable estimate with potentially fewer simulations required.

The algorithm follows the same geometric Brownian motion model as the simple pricer, but for each random variable \( Z \), it also takes \( -Z \) and simulates two price paths. The mean payoff is then calculated using both sets of paths, leading to a more accurate and efficient price estimate.

### Least Squares Monte Carlo Pricer (`LeastSquaresMonteCarloPricer`)

This method is used primarily for American-style options, where the option can be exercised at any time before expiry. It involves simulating multiple price paths and then using regression at each time step to estimate the continuation value of the option.

The continuation value is compared to the intrinsic value (the payoff if exercised at that time), and the option is considered for exercise if the intrinsic value is higher. This process is repeated at each time step, moving backwards from expiry to the present.

The least squares method helps to approximate the optimal exercise strategy for American options, thus providing a fair price estimate that accounts for the option's flexibility.

### Mathematical Efficiency and Effectiveness

The efficiency of these methods is due to the use of vectorized operations in NumPy, which allows for rapid computation of multiple scenarios simultaneously. Random number generation, path simulations, and payoff calculations are performed in bulk, significantly speeding up the process.

The effectiveness stems from the law of large numbers, where increasing the number of simulated paths tends to converge towards the true option price. Moreover, techniques like antithetic variates and regression in the least squares method help to reduce variance and bias in the estimates, leading to more reliable pricing.

These algorithms embody the complexity and adaptability required to estimate option prices in a dynamic financial environment. While they provide robust theoretical estimates, users should be aware of market conditions, model limitations, and the assumptions inherent in the Monte Carlo simulation methods.

---

## Contributing
Contributions to the MCOP are welcome. Please feel free to fork the repository, make changes, and submit pull requests. If you find any issues or have suggestions for improvement, please open an issue in the repository.

## Upcoming Features

This code is actively being developed and there are several exciting features planned for future updates, including:

- Enabling the user to find contracts by a specified range of ask prices.
- Risk measurement features to evaluate accuracy of simulation
- Additional pricing methods for American Options

---
