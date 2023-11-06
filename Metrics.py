import random
import pandas as pd
import time
import yfinance as yf
import itertools

from MCOP import SimpleMCPricer, AntitheticMCPricer, LeastSquaresMonteCarloPricer, Contract


def get_random_option(tickers):
    random_ticker = random.choice(tickers)
    stock = yf.Ticker(random_ticker)
    options = stock.options
    random_date = random.choice(options)
    option_chain = stock.option_chain(random_date)
    all_options = pd.concat([option_chain.calls, option_chain.puts])
    random_option = all_options.sample(n=1).iloc[0]
    random_option['date'] = random_date
    random_option['ticker'] = random_ticker
    random_option['spot'] = stock.history(period='1d')['Close'].iloc[-1]
    # print("Option: ")
    # print(random_option[['ticker', 'date', 'strike', 'spot']])
    print(random_ticker)
    # print()
    return random_option


def run_pricer(pricer_class, contract, paths):
    pricer = pricer_class(contract, paths=paths)
    start_time = time.time()
    result = pricer.get_result()
    end_time = time.time()
    # print(result)
    return end_time - start_time


def collect_data(pricer_paths_dict, tickers, num_runs=1):
    runtimes = pd.DataFrame(columns=['Pricer', 'Paths', 'Runtime'])
    averages = {}  # A dictionary to hold the sum of runtimes and counts
    total_runs = 0

    # Initialize averages dictionary
    for pricer_class in pricer_paths_dict.keys():
        averages[pricer_class.__name__] = {str(paths): {'total_time': 0, 'count': 0} for paths in pricer_paths_dict[pricer_class]}

    # Create a list of all paths for each pricer, maintaining the order
    all_paths = list(itertools.chain.from_iterable(pricer_paths_dict.values()))

    for run_count in range(num_runs):
        for paths in all_paths:
            for pricer_class in pricer_paths_dict.keys():
                if paths in pricer_paths_dict[pricer_class]:  # Only run if the pricer uses this number of paths
                    while True:
                        try:
                            contract = Contract(get_random_option(tickers))
                            runtime = run_pricer(pricer_class, contract, paths)
                            break  # If successful, exit the loop
                        except Exception as e:
                            print(f"An error occurred: {e}. Trying next option.")
                            # If an error occurs, it will automatically try the next ticker

                    # Update averages dictionary
                    averages[pricer_class.__name__][str(paths)]['total_time'] += runtime
                    averages[pricer_class.__name__][str(paths)]['count'] += 1
                    total_runs += 1

                    # Calculate new average runtime
                    new_average = averages[pricer_class.__name__][str(paths)]['total_time'] / averages[pricer_class.__name__][str(paths)]['count']
                    print(f"Iteration {run_count+1}, {pricer_class.__name__} with {paths:,} paths: Runtime {runtime:.2f}s (Running Average: {new_average:.2f}s)")
                    print('----------------------------------------------------------------')

    # Prepare final averages for all pricer and path combinations
    final_averages = []
    for pricer, paths_data in averages.items():
        for paths, data in paths_data.items():
            if data['count'] > 0:  # Avoid division by zero
                average_runtime = data['total_time'] / data['count']
                final_averages.append({
                    'Pricer': pricer,
                    'Paths': paths,
                    'Average Runtime': average_runtime
                })

    return total_runs, pd.DataFrame(final_averages)


if __name__ == '__main__':
    pricer_paths_dict = {
        SimpleMCPricer: [1000000, 10000000, 100000000],
        AntitheticMCPricer: [1000000, 10000000, 100000000],
        LeastSquaresMonteCarloPricer: [10000, 100000, 1000000]
    }

    # List of S&P 500 tickers
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)
    sp500_df = table[0]
    sp500_tickers = sp500_df['Symbol'].tolist()
    # print(sp500_tickers)

    sims = 180
    start = time.time()
    total_runs, runtimes_data = collect_data(pricer_paths_dict, sp500_tickers, num_runs=sims)
    elapsed_seconds = time.time() - start
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    elapsed_time = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print('---------------------------------------------------------------------------------------------')
    print('Final Averages: ')
    print(f'{total_runs} simulations ran.')
    print(f'Elapsed Time: {elapsed_time}')
    print()
    print(runtimes_data)
    print('---------------------------------------------------------------------------------------------')

    """
        ---------------------------------------------------------------------------------------------
        Final Averages: 
        3420 simulations ran.
        Elapsed Time: 2h 58m 35s
        
                                 Pricer      Paths  Average Runtime
        0                SimpleMCPricer    1000000         0.036074
        1                SimpleMCPricer   10000000         0.395684
        2                SimpleMCPricer  100000000         4.034840
        3            AntitheticMCPricer    1000000         0.028900
        4            AntitheticMCPricer   10000000         0.310815
        5            AntitheticMCPricer  100000000         3.166422
        6  LeastSquaresMonteCarloPricer      10000         0.103871
        7  LeastSquaresMonteCarloPricer     100000         0.914739
        8  LeastSquaresMonteCarloPricer    1000000        10.397005
        ---------------------------------------------------------------------------------------------
        
        Process finished with exit code 0
    """
