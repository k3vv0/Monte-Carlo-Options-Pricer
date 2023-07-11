import keyboard
import math
import numpy as np
import pandas as pd
import time
import Util
import yfinance as yf


class Book:
    def __init__(self, df, rpp=25):
        self.df = df
        self.results_per_page = rpp
        self.page = 0
        self.total_rows = df.shape[0]
        self.num_chunks = self.total_rows // self.results_per_page
        if self.total_rows % self.results_per_page != 0:
            self.num_chunks += 1

    def getChoice(self):
        choice = -1
        while choice < 0:
            start = self.page * self.results_per_page
            end = min((self.page + 1) * self.results_per_page, self.total_rows)
            print(self.df.iloc[start:end])
            print(f"\nPage {self.page + 1} of {self.num_chunks} showing results {start + 1} to {end}.")
            print("\nEnter selection or navigate pages {< ~ prev, > ~ next}:")
            print("> ")
            # wait for any key event
            event = keyboard.read_event(suppress=True)
            # check if the event is a key down event (ignore key up events)
            if event.event_type == 'down':
                # ">" key: move to next page
                if event.name == 'right':
                    self.page = (self.page + 1) % self.num_chunks
                    print("Next page. Current page: ", self.page)

                # "<" key: move to previous page
                elif event.name == 'left':
                    self.page = (self.page - 1) % self.num_chunks
                    self.page = abs(self.page)
                    print("Previous page. Current page: ", self.page)

                # any other key: take in as input
                else:
                    choice = int(input())
        return choice


class Contract:
    def __init__(self, info):
        self.info = info
        self.strike = float(info[['strike']].iloc[0])
        self.vol = float(info[['impliedVolatility']].iloc[0])
        self.expiry = float(Util.calcDaysBetween(info[['date']]))
        self.expiry /= 365
        self.spot = float(info[['spot']].iloc[0])

    def print(self):
        print(self.info)

    def printVitals(self):
        print(f"Strike: ${self.strike}")
        print(f"Spot: ${self.spot}")
        print(f"Expiry: {self.expiry} years")
        print(f"Volatility: {self.vol}")


class Option:
    def __init__(self, tick, call=True):
        self.ticker = tick.upper()
        self.sec = yf.Ticker(tick)
        self.spot = self.sec.info['regularMarketPreviousClose']
        self.dates = self.sec.options
        self.call = call

    # Returns all contracts in a given date range
    def contractByDate(self, start_date, end_date):
        df = pd.DataFrame()
        for date in self.dates:
            if date < start_date:
                continue
            elif date > end_date:
                continue
            else:
                contracts = pd.DataFrame()
                chain = self.sec.option_chain(date)
                if self.call:
                    contracts = chain.calls
                else:
                    contracts = chain.puts
                contracts[['date']] = date
                contracts[['ticker']] = self.ticker
                contracts[['spot']] = self.spot
                contracts = contracts[['date', 'strike', 'lastPrice', 'impliedVolatility', 'ticker', 'spot']]
                # print(contracts)
                df = pd.concat([df, contracts], ignore_index=True)
        print(f"{df.shape[0]} contracts found for '{self.ticker}' expiring between {start_date} and {end_date}.")
        results = Book(df)
        choice = results.getChoice()
        row = df.iloc[choice]
        row = row[['ticker', 'date', 'strike', 'spot', 'lastPrice', 'impliedVolatility']]
        return Contract(row)

    # Returns all contracts in a given strike price range
    def contractByStrike(self, low, high):
        df = pd.DataFrame()
        for date in self.dates:
            contracts = pd.DataFrame()
            chain = self.sec.option_chain(date)
            if self.call:
                contracts = chain.calls
            else:
                contracts = chain.puts
            contracts[['date']] = date
            contracts[['ticker']] = self.ticker
            contracts[['spot']] = self.spot
            contracts = contracts[['date', 'strike', 'lastPrice', 'impliedVolatility', 'ticker', 'spot']]
            contracts = contracts.loc[(contracts['strike'] >= low) & (contracts['strike'] <= high)]
            # print(contracts)
            df = pd.concat([df, contracts], ignore_index=True)
        print(f"{df.shape[0]} contracts found for '{self.ticker}' strike price between ${low} and ${high}.")
        df.sort_values(by=['strike', 'date'], inplace=True)
        df = df.reset_index(drop=True)
        results = Book(df)
        choice = results.getChoice()
        row = df.iloc[choice]
        row = row[['ticker', 'date', 'strike', 'spot', 'lastPrice', 'impliedVolatility']]
        return Contract(row)

    # Returns all contracts in a given price range
    def contractByPrice(self, low, high):
        df = pd.DataFrame()
        for date in self.dates:
            contracts = pd.DataFrame()
            chain = self.sec.option_chain(date)
            if self.call:
                contracts = chain.calls
            else:
                contracts = chain.puts
            contracts[['date']] = date
            contracts[['ticker']] = self.ticker
            contracts[['spot']] = self.spot
            contracts = contracts[['date', 'strike', 'lastPrice', 'impliedVolatility', 'ticker', 'spot']]
            contracts = contracts.loc[(contracts['lastPrice'] >= low) & (contracts['lastPrice'] <= high)]
            # print(contracts)
            df = pd.concat([df, contracts], ignore_index=True)
        print(f"{df.shape[0]} contracts found for '{self.ticker}' last price between ${low} and ${high}.")
        df.sort_values(by=['lastPrice', 'date'], inplace=True)
        df = df.reset_index(drop=True)
        results = Book(df)
        choice = results.getChoice()
        row = df.iloc[choice]
        row = row[['ticker', 'date', 'strike', 'spot', 'lastPrice', 'impliedVolatility']]
        return Contract(row)


# Simple Monte Carlo Pricing Class for Vanilla Call Option
class SimpleMCPricer:
    def __init__(self, contract, paths=100000000):
        start = time.time()
        expiry = contract.expiry
        strike = contract.strike
        spot = contract.spot
        vol = contract.vol
        r = 0.05  # risk-free rate

        # The sigma value on the left side of the exponent
        variance = vol ** 2 * expiry
        # The sigma value on the right side of the e exponent
        root_variance = math.sqrt(variance)
        # Corresponds to the (-1/2 * sigma^2)
        ito_corr = -0.5 * variance
        # Corresponds to S0e^(rT - 1/2 sigma^2T)
        moved_spot = spot * math.exp(r * expiry + ito_corr)

        # Simulate for all paths
        gauss_rvs = np.random.normal(size=paths)
        spot_paths = moved_spot * np.exp(root_variance * gauss_rvs)
        payoffs = np.maximum(spot_paths - strike, 0)

        self.mean = np.mean(payoffs) * math.exp(-r * expiry)
        self.std_error = np.std(payoffs) / np.sqrt(paths) * math.exp(-r * expiry)
        self.time = time.time()-start

    def getMean(self):
        return round(self.mean, 2)

    def getStdError(self):
        return round(self.std_error, 2)


class AntitheticMCPricer:
    def __init__(self, contract, paths=100000000):
        start = time.time()
        expiry = contract.expiry
        strike = contract.strike
        spot = contract.spot
        vol = contract.vol
        r = 0.05  # risk-free rate

        variance = vol ** 2 * expiry
        root_Variance = math.sqrt(variance)
        itoCorr = -0.5 * variance
        movedSpot = spot * math.exp(r * expiry + itoCorr)

        # Simulate for all paths
        paths //= 2  # since we will use each random variable twice
        gauss_rvs = np.random.normal(size=paths)
        gauss_rvs = np.concatenate((gauss_rvs, -gauss_rvs))  # Antithetic variates

        spotPaths = movedSpot * np.exp(root_Variance * gauss_rvs)
        payoffs = np.maximum(spotPaths - strike, 0)

        self.mean = np.mean(payoffs) * math.exp(-r * expiry)
        self.std_error = np.std(payoffs) / np.sqrt(paths * 2) * math.exp(-r * expiry)
        self.time = time.time()-start

    def getMean(self):
        return round(self.mean, 2)

    def getStdError(self):
        return round(self.std_error, 2)


# Set the display options
Util.openWindow()

# Sample run
ticker = Option(input("Enter a ticker: "))
contract = ticker.contractByDate('2023-07-21', '2023-08-21')
model1 = SimpleMCPricer(contract)
model2 = AntitheticMCPricer(contract)
# print("Fair price: ", model.getMean())
# contract = ticker.contractByStrike(45, 55)
# model = SimpleMCPricer(contract)
# contract = ticker.contractByPrice(20, 50)
# model = SimpleMCPricer(contract)
print("Simple Fair price: ", model1.getMean())
print("Simple Std Error: ", model1.getStdError())
print("Simple Time: ", model1.time, "\n")
print("Antithetic Fair Price: ", model2.getMean())
print("Antithetic Std Error: ", model2.getStdError())
print("Antithetic Total Time: ", model2.time)
