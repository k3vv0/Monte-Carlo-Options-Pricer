import keyboard
import pandas as pd
import time
import Util
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression


class Book:
    def __init__(self, df, rpp=10):
        self.df = df
        self.results_per_page = rpp
        self.page = 0
        self.total_rows = df.shape[0]
        self.num_chunks = self.total_rows // self.results_per_page
        if self.total_rows % self.results_per_page != 0:
            self.num_chunks += 1

    def getChoice(self):
        choice = -1
        navigated = True
        while choice < 0:
            # Only compute start and end if choice is still -1
            if navigated:
                navigated = False  # Flag to track navigation event
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
                    print("Next page. Current page: ", self.page + 1)
                    navigated = True  # Set navigated to True

                # "<" key: move to previous page
                elif event.name == 'left':
                    self.page = (self.page - 1) % self.num_chunks
                    self.page = abs(self.page)
                    print("Previous page. Current page: ", self.page + 1)
                    navigated = True  # Set navigated to True

                # any other key: take in as input
                else:
                    try:
                        choice = int(input())
                        # choice = int(event.name)
                    except ValueError:
                        pass

        return choice


class Contract:
    def __init__(self, info, call=True):
        self.info = info
        self.lastPrice = info['lastPrice']
        self.strike = float(info[['strike']].iloc[0])
        self.vol = float(info[['impliedVolatility']].iloc[0])
        self.expiry = float(Util.calcDaysBetween(info['date']))
        self.expiry /= 252
        self.spot = float(info[['spot']].iloc[0])
        self.call = call

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
        self.spot = self.sec.history(period='1d')['Close'].iloc[-1]
        self.dates = self.sec.options
        self.call = call

    def contractByDate(self):
        dates = pd.DataFrame(self.dates, columns=['date'])
        print(f"{len(dates)} expiry dates found for '{self.ticker}' options contracts.")
        date_book = Book(dates)
        date = date_book.getChoice()
        row = dates.iloc[date]
        row = row[['date']]
        contract_date = row[['date']].iloc[0]
        chain = self.sec.option_chain(contract_date)
        contracts = pd.DataFrame()
        df = pd.DataFrame()
        if self.call:
            contracts = chain.calls
        else:
            contracts = chain.puts
        contracts[['date']] = dates.iloc[date]
        contracts[['ticker']] = self.ticker
        contracts[['spot']] = self.spot
        contracts = contracts[['date', 'strike', 'lastPrice', 'impliedVolatility', 'ticker', 'spot']]
        # print(contracts)
        df = pd.concat([df, contracts], ignore_index=True)

        print(f"{df.shape[0]} contracts found for '{self.ticker}' expiring on {contract_date}.")
        df.sort_values(by=['strike', 'date'], inplace=True)
        df = df.reset_index(drop=True)
        results = Book(df)
        choice = results.getChoice()
        row = df.iloc[choice]
        row = row[['ticker', 'date', 'strike', 'spot', 'lastPrice', 'impliedVolatility']]
        return Contract(info=row, call=self.call)



    # Returns all contracts in a given date range
    def contractByDateRange(self, start_date, end_date):
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
        df.sort_values(by=['strike', 'date'], inplace=True)
        df = df.reset_index(drop=True)
        results = Book(df)
        choice = results.getChoice()
        row = df.iloc[choice]
        row = row[['ticker', 'date', 'strike', 'spot', 'lastPrice', 'impliedVolatility']]
        return Contract(info=row, call=self.call)

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
        return Contract(info=row, call=self.call)

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
        return Contract(info=row, call=self.call)


# Simple Monte Carlo Pricing Class for Vanilla Call Option
class SimpleMCPricer:
    def __init__(self, contract, paths=1000000, rfr=0.05):
        self.contract = contract
        self.paths = paths
        self.r = rfr
        self.mean = None
        self.std_error = None
        self.confidence_interval = None
        self.time_taken = None

    def simulate(self):
        start = time.time()
        expiry = self.contract.expiry
        strike = self.contract.strike
        spot = self.contract.spot
        vol = self.contract.vol

        # Vectorized simulation
        # print(f"Volatility: {vol}")
        # print(f"Expiry: {expiry}")
        # print(f"Spot: {spot}")
        gauss_rvs = np.random.normal(size=self.paths)
        # print(f"First 10 random variables: {gauss_rvs[:10]}")
        spot_paths = spot * np.exp((self.r - 0.5 * vol ** 2) * expiry + vol * np.sqrt(expiry) * gauss_rvs)
        # print(f"First 10 spot paths: {spot_paths[:10]}")
        payoffs = np.maximum(spot_paths - strike, 0)
        # print(f"First 10 payoffs: {payoffs[:10]}")

        # Calculating statistics
        discounted_payoffs = payoffs * np.exp(-self.r * expiry)
        self.mean = np.mean(discounted_payoffs)
        self.std_error = np.std(discounted_payoffs) / np.sqrt(self.paths)
        conf_margin = 1.96 * self.std_error  # 95% confidence interval
        self.confidence_interval = (self.mean - conf_margin, self.mean + conf_margin)
        self.time_taken = time.time() - start

    def get_result(self):
        if self.mean is None:
            self.simulate()
        return {
            f'paths walked: {self.paths:,}',
            f'mean: {round(self.mean, 2)}',
            f'std_error: {round(self.std_error, 2)}',
            f'confidence_interval: {tuple(map(lambda x: round(x, 2), self.confidence_interval))}',
            f'time_taken: {round(self.time_taken, 2)}'
        }


class AntitheticMCPricer(SimpleMCPricer):
    def simulate(self):
        start = time.time()
        expiry = self.contract.expiry
        strike = self.contract.strike
        spot = self.contract.spot
        vol = self.contract.vol

        # Preallocate arrays for payoffs
        payoffs = np.zeros(self.paths)  # Antithetic paths are included so no need to halve the size

        # Generate random variables for half the paths
        gauss_rvs = np.random.normal(size=self.paths // 2)
        antithetic_rvs = -gauss_rvs  # Antithetic variates

        # Calculate spot paths using vectorization
        spot_paths = spot * np.exp((self.r - 0.5 * vol ** 2) * expiry + vol * np.sqrt(expiry) * gauss_rvs)
        antithetic_paths = spot * np.exp((self.r - 0.5 * vol ** 2) * expiry + vol * np.sqrt(expiry) * antithetic_rvs)

        # Combine and calculate payoffs in one step
        payoffs[:self.paths // 2] = np.maximum(spot_paths - strike, 0)
        payoffs[self.paths // 2:] = np.maximum(antithetic_paths - strike, 0)

        # Calculate statistics without loop
        self.mean = np.mean(payoffs) * np.exp(-self.r * expiry)
        self.std_error = np.std(payoffs) / np.sqrt(self.paths) * np.exp(-self.r * expiry)
        conf_margin = 1.96 * self.std_error  # 95% confidence interval
        self.confidence_interval = (self.mean - conf_margin, self.mean + conf_margin)
        self.time_taken = time.time() - start


class LeastSquaresMonteCarloPricer:
    def __init__(self, contract, rfr=0.05, paths=10000, time_steps=100):
        self.contract = contract
        self.risk_free_rate = rfr
        self.paths = paths
        self.time_steps = time_steps
        self.dt = contract.expiry / time_steps
        self.disc = np.exp(-self.risk_free_rate * self.dt)
        self.payoffs = np.zeros((paths, time_steps))
        self.price = None
        self.std_error = None
        self.time_taken = None
        self.confidence_interval = None

    def simulate_price_paths(self):
        S0 = self.contract.spot
        sigma = self.contract.vol
        dt = self.dt
        paths = self.paths
        time_steps = self.time_steps

        prices = np.zeros((paths, time_steps))
        prices[:, 0] = S0
        Z = np.random.standard_normal((paths, time_steps - 1))
        drift = (self.risk_free_rate - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        # Using cumulative sum of log-returns for vectorization
        log_returns = drift + diffusion * Z
        prices[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
        return prices

    def calculate_payoffs(self, prices):
        K = self.contract.strike

        if self.contract.call:
            payoffs = np.maximum(prices - K, 0)
        else:
            payoffs = np.maximum(K - prices, 0)
        # print(payoffs)
        return payoffs

    def estimate_continuation_value(self, prices, intrinsic_values, time_step):
        regression = LinearRegression()
        in_the_money = intrinsic_values > 0
        if not np.any(in_the_money):  # If no paths are in the money, return zeros
            return np.zeros_like(prices)

        X = prices[in_the_money]
        Y = intrinsic_values[in_the_money] * self.disc
        regression.fit(X.reshape(-1, 1), Y)

        continuation_values = np.zeros_like(prices)
        continuation_values[in_the_money] = regression.predict(X.reshape(-1, 1))
        return continuation_values

    def get_price(self):
        start = time.time()
        prices = self.simulate_price_paths()
        intrinsic_values = self.calculate_payoffs(prices)

        for t in reversed(range(self.time_steps - 1)):
            continuation_values = self.estimate_continuation_value(prices[:, t], intrinsic_values[:, t + 1], t)
            exercise = intrinsic_values[:, t] > continuation_values
            # No need to loop through all paths; just use boolean indexing
            intrinsic_values[exercise, t + 1:] = 0
            intrinsic_values[exercise, t] = intrinsic_values[exercise, t]

        # Using boolean indexing to discount only exercised paths
        exercised = intrinsic_values > 0
        exercise_times = exercised.argmax(axis=1)
        discounted_payoff = np.zeros_like(exercise_times, dtype=float)

        for path in range(self.paths):
            if exercised[path, exercise_times[path]]:
                discounted_payoff[path] = intrinsic_values[path, exercise_times[path]] * self.disc ** exercise_times[
                    path]

        self.price = np.mean(discounted_payoff)
        self.std_error = np.std(discounted_payoff) / np.sqrt(self.paths)
        conf_margin = 1.96 * self.std_error
        self.confidence_interval = (self.price - conf_margin, self.price + conf_margin)
        self.time_taken = time.time() - start

        return self.price, self.std_error

    def get_result(self):
        if self.price is None:
            self.get_price()
        return {
            f'paths walked: {self.paths:,}',
            f'mean: {round(self.price, 2)}',
            f'std_error: {round(self.std_error, 2)}',
            f'confidence_interval: {tuple(map(lambda x: round(x, 2), self.confidence_interval))}',
            f'time_taken: {round(self.time_taken, 2)}'
        }


if __name__ == '__main__':
    rate = Util.get_risk_free_rate(duration='1m')
    # Set the display options
    Util.openWindow()

    # Sample run
    ticker = Option(input("Enter a ticker: "))
    start, end = Util.definePeriod()
    contract = ticker.contractByDate()
    print(f'Listed Contract Price: {round(contract.lastPrice, 2)}')
    print()
    print("Calculating Fair Price...")
    print('------------------------------------------------------------------')
    print('Vanilla/European Options')
    print()
    print("Simple:")
    """
    model1 = SimpleMCPricer(contract, paths=10, rfr=rate)
    result = model1.get_result()
    for r in result:
        print(r)
    print()
    model2 = SimpleMCPricer(contract, paths=10000, rfr=rate)
    result = model2.get_result()
    for r in result:
        print(r)
    print()
    """
    model1 = SimpleMCPricer(contract, paths=100000000, rfr=rate)
    result = model1.get_result()
    for r in result:
        print(r)
    print()
    print("Antithetic:")
    model2 = AntitheticMCPricer(contract, paths=100000000, rfr=rate)
    result = model2.get_result()
    for r in result:
        print(r)
    print()
    print('------------------------------------------------------------------')
    print('American Options')
    print()
    print('Least Squares:')
    model3 = LeastSquaresMonteCarloPricer(contract, paths=1000000, rfr=rate)
    result = model3.get_result()
    for r in result:
        print(r)
