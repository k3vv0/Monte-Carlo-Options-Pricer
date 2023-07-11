import keyboard
import math
import numpy as np
import pandas as pd
import yfinance as yf
import Util


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
        print("Find contract by strike price... coming soon!")

    # Returns all contracts in a given ask price range
    def contractByAsk(self, low, high):
        print("Find contract by last price... coming soon!")


# Simple Monte Carlo Pricing Class for Vanilla Call Option
class SimpleMCPricer:
    def __init__(self, contract, paths=1000000):
        expiry = contract.expiry
        strike = contract.strike
        spot = contract.spot
        vol = contract.vol
        # Calculate risk-free rate... coming soon!
        r = 0.05
        # The sigma value on the left side of the exponent
        self.variance = vol ** 2 * expiry
        # The sigma value on the right side of the e exponent
        self.root_Variance = math.sqrt(self.variance)
        # Corresponds to the (-1/2 * sigma^2)
        self.itoCorr = -0.5 * self.variance
        # Corresponds to S0e^(rT - 1/2 sigma^2T)
        self.movedSpot = spot * math.exp(r * expiry + self.itoCorr)
        self.runningSum = 0
        # Simulate for all paths
        for i in range(0, paths):
            thisGauss = np.random.normal()
            # Our rootVariance already has been multiplied by the expiry
            thisSpot = self.movedSpot * math.exp(self.root_Variance * thisGauss)
            # Determine payoff of this specific path
            thisPayoff = thisSpot - strike
            # Value of option is zero is our price is less than the strike
            thisPayoff = thisPayoff if thisPayoff > 0 else 0
            self.runningSum += thisPayoff

        self.mean = self.runningSum / paths
        self.mean *= math.exp(-r * expiry)

    def getMean(self):
        return round(self.mean, 2)


# Set the display options
Util.openWindow()

# Sample run
ticker = Option(input("Enter a ticker: "))
contract = ticker.contractByDate('2024-07-21', '2025-08-11')
model = SimpleMCPricer(contract)
print(model.getMean())
