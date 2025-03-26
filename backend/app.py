from flask import Flask, request, jsonify
from flask_cors import CORS
import osmnx as ox
import networkx as nx
from collections import deque
import os
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

# Global variables to store graph and active route
G = None
active_route = None
obstacles = []

def initialize_graph(location='Amrita Campus, Kerala, India'):
    global G
    try:
        # Try with place name first
        print(f"Attempting to download graph for: {location}")
        G = ox.graph_from_place(location, network_type='all')
    except Exception as e:
        print(f"Error with place name: {e}")
        try:
            # Fall back to bounding box specific to Amrita campus
            print("Using bounding box for Amrita campus")
            # Fixed graph_from_bbox function call
            north, south, east, west = 10.91, 10.89, 76.91, 76.89  # Amrita campus bounding box
            G = ox.graph.graph_from_bbox(north, south, east, west, network_type='all')
        except Exception as e2:
            print(f"Error with bounding box: {e2}")
            # If all else fails, create a simple grid network
            print("Creating synthetic grid network")
            G = create_campus_grid()
    
    return G

def create_campus_grid():
    try:
        # Try to get the actual map data
        print("Attempting to download map data via OSMnx...")
        try:
            # First try to get data via geocoder
            campus_graph = ox.graph_from_place("Amrita University Campus, Kerala, India", network_type="drive")
            print("Successfully obtained map via geocoder")
        except Exception as e:
            print(f"Geocoder failed: {e}. Falling back to bounding box method...")
            # If geocoder fails, try bounding box for Amrita Campus area (Kerala, India)
            # Coordinates for the Amrita Vishwa Vidyapeetham campus area
            north, south, east, west = 10.9047, 10.8947, 76.9086, 76.8986
            try:
                # Try to get the graph from the bounding box
                campus_graph = ox.graph_from_bbox(north, south, east, west, network_type="drive")
                print("Successfully obtained map via bounding box")
            except Exception as bbox_e:
                print(f"Bounding box method failed: {bbox_e}")
                # Create a synthetic grid as a last resort
                print("Creating synthetic campus grid...")
                # Create a synthetic grid graph
                campus_graph = nx.grid_graph(dim=[5, 5])
                # Convert to a directed graph
                campus_graph = nx.DiGraph(campus_graph)
                # Add attributes to edges
                for u, v in campus_graph.edges():
                    campus_graph[u][v]['length'] = 100  # meters
                    campus_graph[u][v]['name'] = f"Road_{u}_{v}"
        
        # Define important campus landmarks as nodes
        landmarks = {
            "main_gate": (0, 0),
            "admin_block": (1, 1),
            "academic_block": (2, 2),
            "library": (3, 1),
            "hostel": (4, 0),
            "cafeteria": (2, 3),
            "sports_ground": (1, 4),
            "parking": (0, 3),
            "auditorium": (3, 4),
            "lab_block": (4, 2)
        }
        
        # Define main roads that connect different parts of the campus
        main_roads = {
            "north_south_1": [(0, i) for i in range(5)],  # Vertical road on the west side
            "north_south_2": [(4, i) for i in range(5)],  # Vertical road on the east side
            "east_west_1": [(i, 0) for i in range(5)],    # Horizontal road on the south side
            "east_west_2": [(i, 4) for i in range(5)]     # Horizontal road on the north side
        }
        
        # Create a mapping of nodes to their coordinates for easy access
        node_map = {}
        for landmark, pos in landmarks.items():
            node_map[landmark] = pos
        
        for road_name, nodes in main_roads.items():
            for i, node in enumerate(nodes):
                road_node_name = f"{road_name}_{i}"
                node_map[road_node_name] = node
        
        # Function to connect landmarks to the nearest main road
        def connect_to_main_road(landmark, position):
            # Find the closest main road node
            min_dist = float('inf')
            closest_road_node = None
            
            for road_name, nodes in main_roads.items():
                for i, node in enumerate(nodes):
                    road_node_name = f"{road_name}_{i}"
                    # Make sure the node exists in our map
                    if road_node_name not in node_map:
                        continue
                        
                    dist = manhattan_distance(position, node)
                    if dist < min_dist:
                        min_dist = dist
                        closest_road_node = road_node_name
            
            # Return the closest road node
            return closest_road_node
        
        # Build our simplified roadmap graph
        roadmap = nx.DiGraph()
        
        # Add landmark nodes
        for landmark, pos in landmarks.items():
            roadmap.add_node(landmark, pos=pos)
        
        # Add main road nodes
        for road_name, nodes in main_roads.items():
            for i, node in enumerate(nodes):
                road_node_name = f"{road_name}_{i}"
                roadmap.add_node(road_node_name, pos=node)
        
        # Connect consecutive nodes on each main road
        for road_name, nodes in main_roads.items():
            for i in range(len(nodes) - 1):
                node1 = f"{road_name}_{i}"
                node2 = f"{road_name}_{i+1}"
                roadmap.add_edge(node1, node2, weight=1)
                roadmap.add_edge(node2, node1, weight=1)  # Add the reverse edge for bidirectional travel
        
        # Connect the ends of perpendicular roads to create a grid
        # Connect north-south roads with east-west roads at their intersections
        for ns_road in ["north_south_1", "north_south_2"]:
            for ew_road in ["east_west_1", "east_west_2"]:
                # Find the nodes that should be connected
                ns_nodes = main_roads[ns_road]
                ew_nodes = main_roads[ew_road]
                
                # Determine the intersection point
                for i, ns_node in enumerate(ns_nodes):
                    for j, ew_node in enumerate(ew_nodes):
                        if ns_node == ew_node:  # This is an intersection
                            ns_node_name = f"{ns_road}_{i}"
                            ew_node_name = f"{ew_road}_{j}"
                            
                            # Ensure nodes exist before adding edges
                            if ns_node_name in node_map and ew_node_name in node_map:
                                roadmap.add_edge(ns_node_name, ew_node_name, weight=0.1)
                                roadmap.add_edge(ew_node_name, ns_node_name, weight=0.1)
        
        # Connect landmarks to the main roads
        for landmark, pos in landmarks.items():
            closest_road_node = connect_to_main_road(landmark, pos)
            if closest_road_node and closest_road_node in roadmap.nodes:
                roadmap.add_edge(landmark, closest_road_node, weight=0.5)
                roadmap.add_edge(closest_road_node, landmark, weight=0.5)
        
        print("Campus grid created successfully")
        return roadmap, node_map, landmarks
        
    except Exception as e:
        print(f"Error creating campus grid: {e}")
        traceback.print_exc()
        # Return a minimal fallback grid
        g = nx.DiGraph()
        nodes = ["main_gate", "academic_block", "hostel", "cafeteria"]
        positions = {
            "main_gate": (0, 0),
            "academic_block": (1, 1),
            "hostel": (0, 2),
            "cafeteria": (2, 1)
        }
        
        for node in nodes:
            g.add_node(node, pos=positions[node])
        
        # Connect them in a loop
        g.add_edge("main_gate", "academic_block", weight=1)
        g.add_edge("academic_block", "cafeteria", weight=1)
        g.add_edge("cafeteria", "hostel", weight=1)
        g.add_edge("hostel", "main_gate", weight=1)
        
        # Add reverse edges for bidirectional travel
        g.add_edge("academic_block", "main_gate", weight=1)
        g.add_edge("cafeteria", "academic_block", weight=1)
        g.add_edge("hostel", "cafeteria", weight=1)
        g.add_edge("main_gate", "hostel", weight=1)
        
        return g, positions, positions

def connect_path_nodes(G, node_map, *node_names):
    """Connect a sequence of nodes to form a path"""
    for i in range(len(node_names) - 1):
        try:
            from_node = node_map[node_names[i]]
            to_node = node_map[node_names[i+1]]
            connect_nodes(G, from_node, to_node)
        except KeyError as e:
            print(f"Warning: Node {e} not found in map. Skipping connection.")

def connect_nodes(G, node1, node2):
    """Connect two nodes with bidirectional edges"""
    # Get coordinates
    y1, x1 = G.nodes[node1]['y'], G.nodes[node1]['x']
    y2, x2 = G.nodes[node2]['y'], G.nodes[node2]['x']
    
    # Calculate distance
    dist = haversine_distance(y1, x1, y2, x2)
    
    # Add edges in both directions
    G.add_edge(node1, node2, length=dist)
    G.add_edge(node2, node1, length=dist)

def find_closest_node(G, coords, candidate_nodes=None):
    """Find the closest node to the given coordinates"""
    lat, lon = coords
    min_dist = float('inf')
    closest_node = None
    
    nodes_to_check = candidate_nodes if candidate_nodes else G.nodes()
    
    for node in nodes_to_check:
        node_lat = G.nodes[node]['y']
        node_lon = G.nodes[node]['x']
        
        dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
        
        if dist < min_dist:
            min_dist = dist
            closest_node = node
    
    return closest_node

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great circle distance between two points in meters"""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371000  # Radius of earth in meters
    return c * r

def find_nearest_node(G, lat, lon):
    print(f"Finding nearest node to {lat}, {lon}")
    try:
        return ox.distance.nearest_nodes(G, lon, lat)
    except Exception as e:
        print(f"Error finding nearest node: {e}")
        # Find the nearest node by manual calculation if osmnx function fails
        min_dist = float('inf')
        nearest_node = None
        
        for node, data in G.nodes(data=True):
            try:
                node_lat = data['y']
                node_lon = data['x']
                dist = ((float(lat) - node_lat) ** 2 + (float(lon) - node_lon) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    nearest_node = node
            except KeyError:
                continue
        
        print(f"Manually found nearest node: {nearest_node} at distance {min_dist}")
        return nearest_node

def compute_path(G, start_coords, end_coords):
    # Find the nearest nodes to the coordinates
    start_node = find_nearest_node(G, start_coords[0], start_coords[1])
    end_node = find_nearest_node(G, end_coords[0], end_coords[1])
    
    print(f"Finding path from node {start_node} to {end_node}")
    
    # Check if nodes were found
    if start_node is None or end_node is None:
        print("Could not find valid nodes")
        return None, None
    
    # Compute the shortest path
    try:
        route = nx.shortest_path(G, start_node, end_node, weight='length')
        route_coords = []
        for node in route:
            y = G.nodes[node]['y']  # latitude
            x = G.nodes[node]['x']  # longitude
            route_coords.append([y, x])
        print(f"Found route with {len(route_coords)} points")
        return route, route_coords
    except nx.NetworkXNoPath:
        print(f"No path found between nodes {start_node} and {end_node}")
        return None, None
    except Exception as e:
        print(f"Error computing path: {e}")
        return None, None

def reroute(G, start_coords, end_coords, obstacle_coords, radius=50):
    # Find the nearest node to the obstacle
    obstacle_node = find_nearest_node(G, obstacle_coords[0], obstacle_coords[1])
    
    # Create a copy of the graph
    G_temp = G.copy()
    
    # Remove the obstacle node and nearby nodes
    nodes_to_remove = [obstacle_node]
    for node in G.nodes():
        if node != obstacle_node:
            # Calculate distance to obstacle
            node_y = G.nodes[node]['y']
            node_x = G.nodes[node]['x']
            obstacle_y = G.nodes[obstacle_node]['y']
            obstacle_x = G.nodes[obstacle_node]['x']
            dist = ((node_y - obstacle_y) ** 2 + (node_x - obstacle_x) ** 2) ** 0.5
            if dist < radius / 111000:  # Convert meters to degrees (approximate)
                nodes_to_remove.append(node)
    
    # Remove the nodes
    for node in nodes_to_remove:
        if node in G_temp:
            G_temp.remove_node(node)
    
    # Find new path
    return compute_path(G_temp, start_coords, end_coords)

@app.route('/initialize', methods=['POST'])
def init_graph():
    data = request.json
    location = data.get('location', 'Amrita Campus, Kerala, India')
    try:
        initialize_graph(location)
        return jsonify({"status": "success", "message": f"Graph initialized for {location}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/path', methods=['POST'])
def get_path():
    global active_route
    data = request.json
    start_lat = data.get('start_lat')
    start_lon = data.get('start_lon')
    end_lat = data.get('end_lat')
    end_lon = data.get('end_lon')
    
    print(f"Path request: from ({start_lat}, {start_lon}) to ({end_lat}, {end_lon})")
    
    if not all([start_lat, start_lon, end_lat, end_lon]):
        return jsonify({"status": "error", "message": "Missing coordinates"}), 400
    
    if G is None:
        print("Graph not initialized. Initializing now.")
        initialize_graph()
    
    try:
        route, route_coords = compute_path(G, (start_lat, start_lon), (end_lat, end_lon))
        if not route:
            print("No path found. Creating direct path with waypoints.")
            
            # Try to create a path with intermediate waypoints through campus
            route_coords = create_campus_path(float(start_lat), float(start_lon), 
                                            float(end_lat), float(end_lon))
        
        active_route = {
            "route": route,
            "start": (start_lat, start_lon),
            "end": (end_lat, end_lon),
            "coordinates": route_coords
        }
        
        return jsonify({
            "status": "success", 
            "path": route_coords,
            "message": "Path found successfully"
        })
    except Exception as e:
        print(f"Error in get_path: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obstacle', methods=['POST'])
def report_obstacle():
    global active_route, obstacles
    data = request.json
    obstacle_lat = data.get('lat')
    obstacle_lon = data.get('lon')
    
    if not all([obstacle_lat, obstacle_lon]):
        return jsonify({"status": "error", "message": "Missing obstacle coordinates"}), 400
    
    if active_route is None:
        return jsonify({"status": "error", "message": "No active route to reroute"}), 400
    
    obstacles.append((obstacle_lat, obstacle_lon))
    
    try:
        new_route, new_coords = reroute(
            G, 
            active_route["start"], 
            active_route["end"], 
            (obstacle_lat, obstacle_lon)
        )
        
        if not new_route:
            return jsonify({"status": "error", "message": "No alternative path found"}), 404
        
        active_route = {
            "route": new_route,
            "start": active_route["start"],
            "end": active_route["end"],
            "coordinates": new_coords
        }
        
        return jsonify({
            "status": "success", 
            "path": new_coords,
            "message": "Path rerouted successfully"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/obstacles', methods=['GET'])
def get_obstacles():
    return jsonify({"obstacles": obstacles})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "graph_loaded": G is not None,
        "active_route": active_route is not None,
        "obstacle_count": len(obstacles)
    })

def create_campus_path(start_lat, start_lon, end_lat, end_lon):
    """Create a path with waypoints through campus landmarks"""
    
    # Known waypoints - central locations likely to be part of any path
    waypoints = [
        (10.903831, 76.899839),  # Arjuna statue (central point)
        (10.903000, 76.901800),  # Central kitchen area
    ]
    
    # If going east-west, use the east-west main road
    if abs(start_lon - end_lon) > abs(start_lat - end_lat):
        print("East-west dominant path")
        waypoints.append((10.903831, 76.900800))  # East-west road point
        
        # If going to eastern side of campus
        if end_lon > 76.900:
            waypoints.append((10.903831, 76.902000))  # Eastern road point
        # If going to western side of campus
        else:
            waypoints.append((10.903831, 76.897500))  # Western road point
    
    # If going north-south, use the north-south main road
    else:
        print("North-south dominant path")
        waypoints.append((10.903000, 76.899839))  # North-south road point
        
        # If going to northern side of campus
        if end_lat > 10.903:
            waypoints.append((10.905000, 76.899839))  # Northern road point
        # If going to southern side of campus
        else:
            waypoints.append((10.901000, 76.899839))  # Southern road point
    
    # Filter waypoints based on path direction to avoid zigzagging
    filtered_waypoints = []
    for wp_lat, wp_lon in waypoints:
        # If waypoint is roughly in the direction of our path, include it
        if is_waypoint_in_direction(start_lat, start_lon, end_lat, end_lon, wp_lat, wp_lon):
            filtered_waypoints.append((wp_lat, wp_lon))
    
    # Create full path: start -> waypoints -> end
    full_path = [(start_lat, start_lon)]
    
    # Sort waypoints by distance from start to end
    sorted_waypoints = sorted(filtered_waypoints, 
                             key=lambda wp: distance_to_line(start_lat, start_lon, 
                                                            end_lat, end_lon, 
                                                            wp[0], wp[1]))
                                                            
    # Add waypoints to path
    full_path.extend(sorted_waypoints)
    
    # Add end point
    full_path.append((end_lat, end_lon))
    
    # Smooth the path by adding interpolated points between waypoints
    smooth_path = []
    for i in range(len(full_path) - 1):
        wp1_lat, wp1_lon = full_path[i]
        wp2_lat, wp2_lon = full_path[i + 1]
        
        # Add first waypoint
        smooth_path.append([wp1_lat, wp1_lon])
        
        # Add interpolated points
        points = 5  # Number of points to add between waypoints
        for j in range(1, points):
            ratio = j / (points + 1)
            interp_lat = wp1_lat * (1 - ratio) + wp2_lat * ratio
            interp_lon = wp1_lon * (1 - ratio) + wp2_lon * ratio
            smooth_path.append([interp_lat, interp_lon])
    
    # Add final endpoint
    smooth_path.append([end_lat, end_lon])
    
    return smooth_path

def is_waypoint_in_direction(start_lat, start_lon, end_lat, end_lon, wp_lat, wp_lon):
    """Check if waypoint is roughly in the direction from start to end"""
    # Vector from start to end
    vec_end_lat = end_lat - start_lat
    vec_end_lon = end_lon - start_lon
    
    # Vector from start to waypoint
    vec_wp_lat = wp_lat - start_lat
    vec_wp_lon = wp_lon - start_lon
    
    # Dot product to check if vectors point in same direction
    dot_product = vec_end_lat * vec_wp_lat + vec_end_lon * vec_wp_lon
    
    # Waypoint should not increase the path distance too much
    direct_dist = haversine_distance(start_lat, start_lon, end_lat, end_lon)
    path_dist = (haversine_distance(start_lat, start_lon, wp_lat, wp_lon) + 
                haversine_distance(wp_lat, wp_lon, end_lat, end_lon))
    
    return dot_product > 0 and path_dist < direct_dist * 1.7

def distance_to_line(x1, y1, x2, y2, x0, y0):
    """Calculate the distance from point (x0,y0) to line through (x1,y1) and (x2,y2)"""
    num = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    den = ((y2-y1)**2 + (x2-x1)**2)**0.5
    return num/den if den != 0 else 0

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two grid positions"""
    x1, y1 = pos1
    x2, y2 = pos2
    return abs(x1 - x2) + abs(y1 - y2)

@app.route('/find_path', methods=['POST'])
def find_path_endpoint():
    try:
        data = request.get_json()
        
        # Create or get the campus grid
        if not hasattr(app, 'campus_grid') or not hasattr(app, 'node_map') or not hasattr(app, 'landmarks'):
            app.campus_grid, app.node_map, app.landmarks = create_campus_grid()
        
        start_location = data.get('start')
        end_location = data.get('end')
        
        # Validate inputs
        if not start_location or not end_location:
            return jsonify({'error': 'Start and end locations are required'}), 400
        
        print(f"Finding path from {start_location} to {end_location}")
        
        # Check if start and end are in our landmarks
        if start_location not in app.landmarks and end_location not in app.landmarks:
            # Find closest landmarks if custom locations provided
            start_coords = data.get('start_coords')
            end_coords = data.get('end_coords')
            
            if start_coords and len(start_coords) == 2:
                start_location = find_closest_landmark(app.landmarks, start_coords)
            
            if end_coords and len(end_coords) == 2:
                end_location = find_closest_landmark(app.landmarks, end_coords)
        
        # Check if landmarks exist in our graph
        if start_location not in app.campus_grid.nodes or end_location not in app.campus_grid.nodes:
            return jsonify({'error': f'One or both locations not found in campus map. Available locations: {list(app.landmarks.keys())}'}), 400
        
        # Find the shortest path
        try:
            path = nx.shortest_path(app.campus_grid, start_location, end_location, weight='weight')
            
            # Extract coordinates for each node in the path
            path_coords = []
            for node in path:
                pos = app.campus_grid.nodes[node].get('pos')
                if pos:
                    # If using grid coordinates, convert to lat/lon
                    # This is a simplified conversion - in a real system you'd use proper geo-coordinates
                    y, x = pos  # Grid coordinates
                    # Convert to lat/lon (simplified)
                    lat = 10.900 + (y * 0.001)  # Base latitude + offset
                    lon = 76.896 + (x * 0.001)  # Base longitude + offset
                    path_coords.append({'lat': lat, 'lng': lon, 'name': node})
            
            # Calculate estimated distance
            distance = 0
            for i in range(len(path) - 1):
                src = path[i]
                dst = path[i+1]
                if 'weight' in app.campus_grid[src][dst]:
                    distance += app.campus_grid[src][dst]['weight'] * 100  # Convert to meters
            
            result = {
                'path': path,
                'coordinates': path_coords,
                'distance': distance,
                'estimated_time': distance / 83.3  # Assuming 5km/h walking speed (83.3m/min)
            }
            
            return jsonify(result)
        
        except nx.NetworkXNoPath:
            return jsonify({'error': 'No path found between the specified locations'}), 404
    
    except Exception as e:
        print(f"Error finding path: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def find_closest_landmark(landmarks, coords):
    """Find the closest landmark to the given coordinates"""
    closest = None
    min_dist = float('inf')
    
    for name, pos in landmarks.items():
        # Calculate distance
        dist = manhattan_distance(coords, pos)
        if dist < min_dist:
            min_dist = dist
            closest = name
    
    return closest

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Initialize the graph at startup
    initialize_graph()
    app.run(host='0.0.0.0', port=port, debug=True) 