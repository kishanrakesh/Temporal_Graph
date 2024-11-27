import dotenv
import os
from neo4j import GraphDatabase
import heapq
import math
# Load environment variables
load_status = dotenv.load_dotenv("Neo4j-8fb4b46a-Created-2024-10-22.txt")
if load_status is False:
    raise RuntimeError('Environment variables not loaded.')

# Connect to Neo4j
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))

class PriorityQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, item):
        heapq.heappush(self.queue, item)
    
    def dequeue(self):
        return heapq.heappop(self.queue) if not self.is_empty() else None
    
    def is_empty(self):
        return len(self.queue) == 0

def compute_fastest_path(driver, source, destination):
    # Initialize priority queue and result set
    Q = PriorityQueue()
    S = set()
    Gt = {}  # Transformed graph (to track visited nodes)
    
    # Enqueue the starting node
    Q.enqueue((source, -float('inf'), 0, None))  # (node, time, length, parent)
    
    while not Q.is_empty():
        current = Q.dequeue()
        current_node, current_time, current_length, current_parent = current
        
        # Fetch edges from the graph database
        query = """
        MATCH (n)-[e:Flight]->(m)
        WHERE n.AIRPORT_NAME = $current_node
        RETURN e.SCHEDULED_DEPARTURE AS start, e.SCHEDULED_ARRIVAL AS end, m.AIRPORT_NAME AS dest
        """
        results = driver.session().run(query, current_node=current_node)
        
        for record in results:
            interval_start = record["start"]
            interval_end = record["end"]
            dest = record["dest"]
            
            if current_time > interval_start:
                continue
            
            # Create transformed graph nodes
            vOut = (current_node, interval_start, current_length + 1, current)
            vIn = (dest, interval_end, current_length + 1, vOut)
            
            # Check if vIn is already in the transformed graph
            if (vIn[0], vIn[1]) in Gt:
                other_vIn = Gt[(vIn[0], vIn[1])]
                if (other_vIn[1] - get_first(other_vIn)[1]) - (vIn[1] - get_first(vIn)[1]) > 0:
                    continue
            
            # Add vIn to the transformed graph
            Gt[(vIn[0], vIn[1])] = vIn
            
            # Check if the destination is reached
            if dest == destination:
                if not S:
                    S.add(vIn)
                else:
                    s = next(iter(S))  # Get any element from S
                    comp =  (s[1] - get_first(s)[1]) - (vIn[1] - get_first(vIn)[1])
                    if comp > 0:
                        S.clear()
                        S.add(vIn)
                    elif comp == 0:
                        S.add(vIn)
                continue
            
            # Add vIn to the priority queue
            Q.enqueue(vIn)
    
    return S

def get_first(node):
    # Base case: If the node has no parent, return its time
    if node[3] is None or math.isinf(node[3][1]) :
        return node
    
    # Recursive case: Traverse back to the parent's time
    return get_first(node[3])

# Main execution
if __name__ == "__main__":
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        driver.verify_connectivity()
        print("Connection established.")

        # Define source and destination airports
        source = "LAX"  # Example source node
        destination = "PBI"  # Example destination node

        # Compute the minimum consecutive paths
        results = compute_fastest_path(driver, source, destination)
        
        # Display the results
        print("Optimal Solutions:")
        for result in results:
            print(result)
