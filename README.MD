# Path Computation Analysis in Neo4j

## Project Overview

This project analyzes different path computation algorithms using a Neo4j graph database. The algorithms include:
- Earliest Arrival Path
- Latest Departure Path
- Shortest Path
- Fastest Path

The project reads airport and flight data, processes it in Neo4j, and executes these algorithms for multiple data limits and airport pairs. Execution times and results are logged, plotted, and displayed in tabular format.

## Features

- Import airport and flight data into a Neo4j graph database.
- Execute various path computation algorithms.
- Log execution times for analysis.
- Generate visualizations and tables of results.

---

## Prerequisites

Before running the project, ensure the following are installed:

1. **Python**: Version 3.x
2. **pip**: Python's package manager

---

## Installation

### Step 1: Clone the Repository
git clone https://github.com/kishanrakesh/Temporal_Graph.git
cd Temporal_Graph

### Step 2: Create and activate a Virtual Environment
python3 -m venv venv  
source ./venv/bin/activate

### Step 3: Install Dependencies
python3 -m pip install matplotlib  
python3 -m pip install pandas  
python3 -m pip install python-dotenv  
python3 -m pip install neo4j  

## Running the Project

Once the setup is complete, execute the following command to run the project:

python3 wrapper.py

This script will:

Import data into the Neo4j database.
Execute path computations for different algorithms.
Log execution times.
Generate plots and tables for analysis.

## Notes

Ensure the Neo4j server is configured and running before importing data.
Update the .env file with your Neo4j credentials (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD).

## Contact

For any questions or support, contact kishanrakesh@gmail.com or raise an issue in the repository.