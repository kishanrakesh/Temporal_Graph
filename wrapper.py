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
import cartopy
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
dotenv.load_dotenv("Neo4j-8fb4b46a-Created-2024-10-22.txt")

def draw_multiple_routes_with_unique_annotations(routes,
    route_labels,
    time_details,
    title = "Flight Routes",
    disp=True,
    save_to_file=False,
    file_name="flight_routes.png"):
    """
    Draws multiple flight routes on the USA map, places rotated airplane icons,
    annotates points uniquely, and adds time details at midpoints.

    :param routes: List of routes where each route is a list of tuples [(lat, lon), ...].
    :param route_labels: List of labels for each coordinate (e.g., city names).
    :param time_details: List of lists of tuples [(departure_time, arrival_time), ...] for each route.
    :title: graph title
    :disp: whether to display the graph window
    :param save_to_file: If True, saves the plot as an image file.
    :param file_name: Name of the file to save the plot.
    """
    # Set up the map
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())  # USA bounds

    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.STATES, edgecolor="gray")
    ax.add_feature(cfeature.LAND, color="lightgray")
    ax.add_feature(cfeature.LAKES, color="aqua")
    ax.add_feature(cfeature.OCEAN, color="aqua")

    # Track annotated points to avoid duplicates
    annotated_points = set()

    # Iterate over all routes
    for route, labels, times in zip(routes, route_labels, time_details):
        lats, lons = zip(*route)
        # Plot the route line
        ax.plot(
            lons,
            lats,
            color="blue",
            linewidth=2,
            # transform=ccrs.Geodetic(),
        )

        # Annotate unique coordinates
        for i, (lat, lon) in enumerate(route):
            if (lat, lon) not in annotated_points:
                label = labels[i]
                ax.text(
                    lon + 1,
                    lat - 1,
                    label,
                    transform=ccrs.PlateCarree(),
                    fontsize=10,
                    color="black" if i not in {0, len(route) - 1} else ("blue" if i == 0 else "green"),
                    bbox=dict(facecolor="white", edgecolor="gray" if i not in {0, len(route) - 1} else ("blue" if i == 0 else "green"), boxstyle="round,pad=0.3"),
                )
                annotated_points.add((lat, lon))

        # Add airplane icons and time details for each segment
        for i in range(len(route) - 1):
            # Calculate the midpoint of the segment
            start_coord = route[i]
            end_coord = route[i + 1]
            mid_lat = (start_coord[0] + end_coord[0]) / 2
            mid_lon = (start_coord[1] + end_coord[1]) / 2

            # Calculate the angle of the segment
            delta_lat = end_coord[0] - start_coord[0]
            delta_lon = end_coord[1] - start_coord[1]
            angle = np.arctan2(delta_lat, delta_lon) * (180 / np.pi)  # Convert to degrees

            # Place the rotated airplane icon at the midpoint
            ax.text(
                mid_lon,
                mid_lat,
                "\u2708",  # Unicode character for an airplane
                transform=ccrs.PlateCarree(),
                fontsize=30,
                color="red",
                ha="center",
                va="center",
                rotation=angle,  # Rotate the airplane icon
                rotation_mode="anchor",  # Rotate relative to the text anchor point
            )

            # Add the time details below the airplane icon
            departure_time, arrival_time = times[i]
            ax.text(
                mid_lon,
                mid_lat - 2,  # Offset slightly below the airplane
                f"{departure_time} → {arrival_time}",
                transform=ccrs.PlateCarree(),
                fontsize=10,
                color="black",
                ha="center",
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"),
            )

    # Add title
    ax.set_title(title, fontsize=16)

    # Save or display the map
    if save_to_file:
        print("Check inside")
        plt.savefig(file_name, dpi=300)

    if disp:
        plt.show()

def extract_flight_path(node):
    """
    Extracts the flight path information recursively from the nested tuple structure.

    Args:
        node (tuple): Nested tuple containing flight path data.

    Returns:
        list: A list of dictionaries containing the flight segments with details.
    """
    path = []
    while node is not None:
        # Extract details for the current node
        airport_code = node[0]
        arrival_time = node[1]
        departure_time = node[3][1] if node[3] is not None else None
        previous_airport_code = node[3][0] if node[3] is not None else None

        # Append the flight segment details to the path
        if previous_airport_code is not None and departure_time is not None:
            path.append({
                "from": previous_airport_code,
                "to": airport_code,
                "departure_time": departure_time,
                "arrival_time": arrival_time,
            })

        # Move to the next node in the structure
        node = node[3][3] if node[3] is not None else None

    # Reverse the path to get the correct order (from source to destination)
    return path[::-1]

def format_for_drawing(flight_path, airport_coordinates):
    """
    Formats the extracted flight path for the drawing function.

    Args:
        flight_path (list): List of dictionaries from `extract_flight_path`.
        airport_coordinates (dict): Dictionary mapping airport codes to coordinates.

    Returns:
        tuple: (route, route_labels, time_details)
    """
    route = []
    route_labels = []
    time_details = []

    for segment in flight_path:
        # Add coordinates and labels
        from_airport = segment["from"]
        to_airport = segment["to"]
        route.append(airport_coordinates[from_airport])

        airports_data = pd.read_csv("data/airports.csv")
        airport_info = airports_data[airports_data['IATA_CODE'] == from_airport].iloc[0]
        # Extract airport details
        arrival_airport = airport_info['CITY']
        route_labels.append(arrival_airport)
        # Add the last destination
        if len(route) == len(flight_path):
            route.append(airport_coordinates[to_airport])

            airport_info = airports_data[airports_data['IATA_CODE'] == to_airport].iloc[0]
            # Extract airport details
            arrival_airport = airport_info['CITY']

            route_labels.append(arrival_airport)

        # Add time details
        time_details.append((segment["departure_time"], segment["arrival_time"]))
        print("Time --------->",time_details)

    return route, route_labels, time_details

# Function to execute a single path computation and return execution time and number of results
def execute_path_function(driver, source, destination, func,names,num_node):
    start_time = time.time()
    results = func(driver, source, destination)
    end_time = time.time()
    print(results)
    if len(results)==0:
        pass
    else:
        res=list(results)
        print(len(res))

        data_split=[]
        for i in res:

          extract=extract_flight_path(i)
          data_split.append(extract)
        routes=[]
        route_labels_l=[]
        time_details_l=[]
        airport_data = pd.read_csv("data/airports.csv")
        airport_coordinates = dict(zip(airport_data['IATA_CODE'], zip(airport_data['LATITUDE'], airport_data['LONGITUDE'])))
        print(data_split)
        for i in data_split:
            # print(i)
            route, route_labels, time_details = format_for_drawing(i, airport_coordinates)

            routes.append(route)
            route_labels_l.append(route_labels)
            time_details_l.append(time_details)
            print(routes)
            print(route_labels_l)
            print(time_details_l)
        draw_multiple_routes_with_unique_annotations(routes, route_labels_l, time_details_l,title=names+"\n"+source+" -> "+destination,disp=False,save_to_file=True,file_name="images/"+names+"_"+source+"_"+destination+"_"+str(num_node)+".png")

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
        data_limits = [1000,2000,3000,4000]

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
                execution_time, num_results = execute_path_function(driver, source, destination, compute_earliest_arrival,"Earliest Arrival",data_limit)
                table_earliest_arrival.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_latest_departure,"Latest Departure",data_limit)
                table_latest_departure.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_shortest_path,"Shortest Path",data_limit)
                table_shortest_path.append({
                    "Pair (Source -> Destination)": f"{source} → {destination}",
                    "Data Limit": data_limit,
                    "Execution Time (seconds)": execution_time,
                    "Number of Results": num_results
                })

                execution_time, num_results = execute_path_function(driver, source, destination, compute_fastest_path,"Fastest Path",data_limit)
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
            "evaluation/earliest_arrival.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_latest_departure if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Latest Departure Path",
            "evaluation/latest_departure.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_shortest_path if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Shortest Path",
            "evaluation/shortest_path.png"
        )
        plot_execution_times(
            data_limits,
            {pair: [entry["Execution Time (seconds)"] for entry in table_fastest_path if entry["Pair (Source -> Destination)"] == f"{pair[0]} → {pair[1]}"] for pair in pairs},
            "Fastest Path",
            "evaluation/fastest_path.png"
        )

        print("Graphs and tables have been saved.")
