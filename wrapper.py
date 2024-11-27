import time
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for table creation
from upload import import_data_to_neo4j
from earliest_arrival import compute_earliest_arrival
from latest_departure import compute_latest_departure
from shortest_path import compute_shortest_path
from fastest_path import compute_fastest_path
from neo4j import GraphDatabase
import os
import dotenv

dotenv.load_dotenv("Neo4j-8fb4b46a-Created-2024-10-22.txt")

# Function to execute a single path computation and return execution time and number of results
def execute_path_function(driver, source, destination, func):
    start_time = time.time()
    results = func(driver, source, destination)
    end_time = time.time()
    print(results)

    # Log execution time and number of results
    execution_time = end_time - start_time
    return execution_time, len(results)

# Function to plot and save results
def plot_execution_times(data_limits, results, title, filename):
    plt.figure(figsize=(10, 6))
    for pair, times in results.items():
        plt.plot(data_limits, times, marker='o', label=f"{pair[0]} → {pair[1]}")

    plt.title(title)
    plt.xlabel("Data Limit")
    plt.ylabel("Execution Time (seconds)")
    plt.legend()
    plt.grid()

    # Save the plot to a file
    plt.savefig(filename)
    plt.close()  # Close the figure to free memory

# Function to create a DataFrame for execution times and display it
def create_execution_table(data, title):
    df = pd.DataFrame(data)
    print(f"\n{title}:\n")
    print(df)
    # Save table to a CSV file
    df.to_csv(f"tables/{title.lower().replace(' ', '_')}.csv", index=False)

# Main execution
if __name__ == "__main__":
    # Define source-destination pairs
    pairs = [("ATL", "CLD"), ("BOS", "HOU"), ("ATL", "AUS"), ("SBN", "ISP")]

    # Establish connection to Neo4j
    URI = os.getenv("NEO4J_URI")
    AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Connection established.")

        # Data limits
        data_limits = [1000, 2000, 3000, 4000]

        # Dictionaries to store execution data
        table_earliest_arrival = []
        table_latest_departure = []
        table_shortest_path = []
        table_fastest_path = []

        # Run the process for each data limit
        for data_limit in data_limits:
            print(f"\nImporting data into Neo4j with limit {data_limit}...")
            import_data_to_neo4j(data_limit)

            for source, destination in pairs:
                print(f"\nRunning path computations for {source} → {destination} with data limit {data_limit}...")

                # Record execution times and number of results for each function
                execution_time, num_results = execute_path_function(driver, source, destination, compute_earliest_arrival)
                table_earliest_arrival.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_latest_departure)
                table_latest_departure.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_shortest_path)
                table_shortest_path.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_fastest_path)
                table_fastest_path.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

        # Create and display tables
        create_execution_table(table_earliest_arrival, "Earliest Arrival Path")
        create_execution_table(table_latest_departure, "Latest Departure Path")
        create_execution_table(table_shortest_path, "Shortest Path")
        create_execution_table(table_fastest_path, "Fastest Path")

        # Plot and save results
        print("\nPlotting execution times...")
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_earliest_arrival if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Earliest Arrival Path",
            "images/earliest_arrival.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_latest_departure if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Latest Departure Path",
            "images/latest_departure.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_shortest_path if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Shortest Path",
            "images/shortest_path.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_fastest_path if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Fastest Path",
            "images/fastest_path.png"
        )

        print("Graphs and tables have been saved.")
