import dotenv
import os
import pandas as pd
from neo4j import GraphDatabase

def import_data_to_neo4j(row_count):
    # Load environment variables
    load_status = dotenv.load_dotenv("Neo4j-8fb4b46a-Created-2024-10-22.txt")
    if load_status is False:
        raise RuntimeError('Environment variables not loaded.')

    # Connect to Neo4j
    URI = os.getenv("NEO4J_URI")
    AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Connection established.")

        # Clear the database (delete all nodes and relationships)
        reset_query = """
        MATCH (n)
        DETACH DELETE n
        """
        with driver.session() as session:
            session.run(reset_query)
            print("Database reset: All nodes and relationships deleted.")

        # Load the CSV data for airports
        airports_df = pd.read_csv('data/airports.csv')

        # Prepare the data as a list of dictionaries for nodeRecords
        node_records = airports_df[['IATA_CODE']].dropna().to_dict('records')

        # Cypher query to import the airport nodes
        create_nodes_query = """
        UNWIND $nodeRecords AS nodeRecord
        WITH *
        WHERE NOT nodeRecord.IATA_CODE IS NULL
        MERGE (n: Airport { AIRPORT_NAME: nodeRecord.IATA_CODE });
        """
        with driver.session() as session:
            session.run(create_nodes_query, nodeRecords=node_records)
            print("Airport nodes imported into Neo4j successfully.")

        # Load the CSV data
        df = pd.read_csv('data/flights1-1.csv')
        df=df.head(row_count)

        # Prepare the data as a list of dictionaries for relRecords
        rel_records = df[['AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL']].to_dict('records')

        # Define the Cypher query
        cypher_query = """
        UNWIND $relRecords AS relRecord
        MATCH (source: `Airport` { `AIRPORT_NAME`: relRecord.`ORIGIN_AIRPORT` })
        MATCH (target: `Airport` { `AIRPORT_NAME`: relRecord.`DESTINATION_AIRPORT` })
        CREATE (source)-[r: `Flight`]->(target)
        SET r.`AIRLINE` = relRecord.`AIRLINE`
        SET r.`FLIGHT_NUMBER` = toInteger(relRecord.`FLIGHT_NUMBER`)
        SET r.`ORIGIN_AIRPORT` = relRecord.`ORIGIN_AIRPORT`
        SET r.`DESTINATION_AIRPORT` = relRecord.`DESTINATION_AIRPORT`
        SET r.`SCHEDULED_DEPARTURE` = toInteger(relRecord.`SCHEDULED_DEPARTURE`)
        SET r.`SCHEDULED_ARRIVAL` = toInteger(relRecord.`SCHEDULED_ARRIVAL`);
        """

        # Execute the Cypher query with the prepared data
        with driver.session() as session:
            session.run(cypher_query, relRecords=rel_records)
            print("Data loaded into Neo4j successfully.")


# Uncomment to run this directly
if __name__ == "__main__":
    import_data_to_neo4j()