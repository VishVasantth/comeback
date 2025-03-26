import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polyline, useMap } from 'react-leaflet';
import axios from 'axios';
import './App.css';

// Config
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5000';
const DETECTION_URL = process.env.REACT_APP_DETECTION_URL || 'http://localhost:5001';
const DEFAULT_CENTER = [10.903831, 76.899839]; // Arjuna Statue at Amrita

// Amrita campus locations
const AMRITA_LOCATIONS = [
  {name: "A1 Staff Quarters", lat: 10.901408, lon: 76.900564},
  {name: "AB1", lat: 10.900501, lon: 76.902866},
  {name: "AB1 Car parking", lat: 10.900806, lon: 76.901861},
  {name: "AB1 Gym", lat: 10.901732, lon: 76.904144},
  {name: "AB2", lat: 10.903632, lon: 76.898394},
  {name: "AB3", lat: 10.906180, lon: 76.897778},
  {name: "AB4 - Amrita School of AI", lat: 10.904236, lon: 76.903576},
  {name: "Adithi Bhavanam", lat: 10.907319, lon: 76.898877},
  {name: "Advanced Multifunctional Materials and Analysis Lab", lat: 10.904150, lon: 76.898912},
  {name: "Aerospace Lab", lat: 10.902235, lon: 76.904414},
  {name: "Agasthya Bhavanam", lat: 10.902492, lon: 76.896217},
  {name: "Agasthya Bhavanam Mess", lat: 10.902944, lon: 76.896219},
  {name: "Amrita Ashram", lat: 10.902068, lon: 76.901058},
  {name: "Amrita Automotive Research and Testing Centre(AARTC)", lat: 10.903807, lon: 76.895610},
  {name: "Amrita Guest House", lat: 10.901419, lon: 76.898799},
  {name: "Amrita ICTS Office", lat: 10.900775, lon: 76.902631},
  {name: "Amrita Kripa Labs(CoE-AMGT)", lat: 10.901223, lon: 76.902384},
  {name: "Amrita Multi Dimensional Data Analytics Lab", lat: 10.900833, lon: 76.902765},
  {name: "Amrita Recycling Centre(ARC)", lat: 10.908921, lon: 76.90192},
  {name: "Amrita School of Business", lat: 10.904433, lon: 76.901833},
  {name: "Amrita School of physical Sciences", lat: 10.903792, lon: 76.898097},
  {name: "Amrita Sewage Treatment Plant", lat: 10.900125, lon: 76.900002},
  {name: "Amritanjali Hall", lat: 10.904666, lon: 76.899220},
  {name: "Amriteshwari Hall", lat: 10.900436, lon: 76.903798},
  {name: "Anokha hub", lat: 10.901236, lon: 76.901742},
  {name: "Anugraha Hall", lat: 10.906226, lon: 76.898032},
  {name: "Arjuna Statue", lat: 10.903831, lon: 76.899839},
  {name: "Ashram Office", lat: 10.902727, lon: 76.901229},
  {name: "Auditorium", lat: 10.904451, lon: 76.902588},
  {name: "B7B Quarters", lat: 10.908074, lon: 76.899355},
  {name: "Basketball Court 1", lat: 10.900774, lon: 76.904054},
  {name: "Basketball Court 2", lat: 10.901147, lon: 76.904080},
  {name: "Bhrigu Bhavanam", lat: 10.905331, lon: 76.904187},
  {name: "Binding Shop", lat: 10.904569, lon: 76.899354}
];

// Helper component to update map view when path changes
function MapUpdater({ path }) {
  const map = useMap();
  
  useEffect(() => {
    if (path && path.length > 0) {
      map.fitBounds(path);
    }
  }, [path, map]);
  
  return null;
}

function App() {
  // State for coordinates
  const [startLat, setStartLat] = useState('10.901408'); // A1 Staff Quarters
  const [startLon, setStartLon] = useState('76.900564');
  const [endLat, setEndLat] = useState('10.904236'); // AB4 - Amrita School of AI
  const [endLon, setEndLon] = useState('76.903576');
  
  // State for location selection
  const [startLocation, setStartLocation] = useState('A1 Staff Quarters');
  const [endLocation, setEndLocation] = useState('AB4 - Amrita School of AI');
  
  // State for path and map
  const [path, setPath] = useState([]);
  const [currentLocation, setCurrentLocation] = useState(DEFAULT_CENTER);
  
  // State for markers and routing
  const [startMarker, setStartMarker] = useState(null);
  const [endMarker, setEndMarker] = useState(null);
  const [waypoints, setWaypoints] = useState([]);
  const [route, setRoute] = useState(null);
  const [rerouting, setRerouting] = useState(false);
  const [error, setError] = useState(null);
  
  // State for detection service
  const [detectionRunning, setDetectionRunning] = useState(false);
  const [objects, setObjects] = useState([]);
  
  // Refs
  const videoRef = useRef(null);
  const frameIntervalRef = useRef(null);
  
  // Function to find path between two points
  const findPath = async (start, end) => {
    try {
      setRerouting(true);
      
      // Set markers
      setStartMarker({ position: { lat: parseFloat(start.lat), lng: parseFloat(start.lng) }, label: 'A' });
      setEndMarker({ position: { lat: parseFloat(end.lat), lng: parseFloat(end.lng) }, label: 'B' });

      console.log(`Finding path from ${JSON.stringify(start)} to ${JSON.stringify(end)}...`);
      
      // Try up to 3 times to get a path from the server
      let attempts = 0;
      let maxAttempts = 3;
      let pathResult = null;
      
      while (attempts < maxAttempts && !pathResult) {
        attempts++;
        console.log(`Attempt ${attempts} to find path...`);
        
        try {
          // Set a timeout of 10 seconds for the fetch
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 10000);
          
          const response = await fetch(`${BACKEND_URL}/find_path`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              start: start.name || 'custom_location',
              end: end.name || 'custom_location',
              start_coords: [start.gridX || 0, start.gridY || 0],
              end_coords: [end.gridX || 0, end.gridY || 0]
            }),
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
          
          if (!response.ok) {
            const errorData = await response.json();
            console.error(`Server error: ${errorData.error}`);
            throw new Error(errorData.error || 'Server error');
          }
          
          pathResult = await response.json();
          console.log("Path found:", pathResult);
        } catch (err) {
          console.error(`Attempt ${attempts} failed:`, err);
          
          // If this was the last attempt, we'll use the fallback
          if (attempts >= maxAttempts) {
            console.log("Using direct path fallback");
          } else {
            // Wait a moment before the next attempt
            await new Promise(resolve => setTimeout(resolve, 1000));
          }
        }
      }
      
      // If we have a path result from the server, use it
      if (pathResult) {
        const pathCoordinates = pathResult.coordinates.map(coord => ({
          lat: parseFloat(coord.lat),
          lng: parseFloat(coord.lng)
        }));
        
        setRoute({
          path: pathCoordinates,
          distance: pathResult.distance,
          time: pathResult.estimated_time
        });
        
        // Add intermediate markers
        setWaypoints(pathResult.coordinates.slice(1, -1).map((coord, idx) => ({
          position: { lat: parseFloat(coord.lat), lng: parseFloat(coord.lng) },
          label: `${idx + 1}`,
          name: coord.name
        })));
      } else {
        // Create a direct path as fallback
        createDirectPath(start, end);
      }
    } catch (error) {
      console.error("Error finding path:", error);
      
      // Create a direct path as fallback
      createDirectPath(start, end);
      
      setError(`Unable to find optimal path. Using direct route: ${error.message}`);
      setTimeout(() => setError(null), 5000);
    } finally {
      setRerouting(false);
    }
  };
  
  // Function to create a direct path between two points
  const createDirectPath = (start, end) => {
    const startPoint = { 
      lat: parseFloat(start.lat), 
      lng: parseFloat(start.lng) 
    };
    
    const endPoint = { 
      lat: parseFloat(end.lat), 
      lng: parseFloat(end.lng) 
    };
    
    // Create a simple path with a few interpolated points
    const pathCoordinates = [
      startPoint,
      {
        lat: startPoint.lat + (endPoint.lat - startPoint.lat) * 0.25,
        lng: startPoint.lng + (endPoint.lng - startPoint.lng) * 0.25
      },
      {
        lat: startPoint.lat + (endPoint.lat - startPoint.lat) * 0.5,
        lng: startPoint.lng + (endPoint.lng - startPoint.lng) * 0.5
      },
      {
        lat: startPoint.lat + (endPoint.lat - startPoint.lat) * 0.75,
        lng: startPoint.lng + (endPoint.lng - startPoint.lng) * 0.75
      },
      endPoint
    ];
    
    // Calculate a rough distance (in meters) using Haversine formula
    const R = 6371e3; // Earth radius in meters
    const φ1 = startPoint.lat * Math.PI/180;
    const φ2 = endPoint.lat * Math.PI/180;
    const Δφ = (endPoint.lat - startPoint.lat) * Math.PI/180;
    const Δλ = (endPoint.lng - startPoint.lng) * Math.PI/180;
    
    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    const distance = R * c;
    
    // Estimate time (5km/h walking speed = 83.3m/min)
    const estimatedTime = distance / 83.3;
    
    setRoute({
      path: pathCoordinates,
      distance: distance,
      time: estimatedTime,
      isDirectPath: true
    });
    
    // Add waypoints for the intermediate points
    setWaypoints(pathCoordinates.slice(1, -1).map((coord, idx) => ({
      position: { lat: coord.lat, lng: coord.lng },
      label: `${idx + 1}`,
      name: `Waypoint ${idx + 1}`
    })));
  };
  
  // Function to handle start location change
  const handleStartLocationChange = (e) => {
    const locationName = e.target.value;
    setStartLocation(locationName);
    
    const location = AMRITA_LOCATIONS.find(loc => loc.name === locationName);
    if (location) {
      setStartLat(location.lat.toString());
      setStartLon(location.lon.toString());
    }
  };
  
  // Function to handle end location change
  const handleEndLocationChange = (e) => {
    const locationName = e.target.value;
    setEndLocation(locationName);
    
    const location = AMRITA_LOCATIONS.find(loc => loc.name === locationName);
    if (location) {
      setEndLat(location.lat.toString());
      setEndLon(location.lon.toString());
    }
  };
  
  // Function to start detection
  const startDetection = async () => {
    try {
      const response = await axios.post(`${DETECTION_URL}/start`);
      if (response.data.status === 'success') {
        setDetectionRunning(true);
        
        // Start fetching video frames
        if (frameIntervalRef.current) {
          clearInterval(frameIntervalRef.current);
        }
        
        frameIntervalRef.current = setInterval(async () => {
          try {
            // Fetch video frame
            const frameResponse = await axios.get(`${DETECTION_URL}/frame`, {
              responseType: 'blob'
            });
            if (videoRef.current) {
              videoRef.current.src = URL.createObjectURL(frameResponse.data);
            }
            
            // Fetch detected objects
            const objectsResponse = await axios.get(`${DETECTION_URL}/objects`);
            setObjects(objectsResponse.data.objects || []);
          } catch (error) {
            console.error('Error fetching video frame:', error);
          }
        }, 100);
      } else {
        alert('Error starting detection: ' + response.data.message);
      }
    } catch (error) {
      console.error('Error starting detection:', error);
      alert('Error starting detection: ' + error.message);
    }
  };
  
  // Function to stop detection
  const stopDetection = async () => {
    try {
      const response = await axios.post(`${DETECTION_URL}/stop`);
      if (response.data.status === 'success') {
        setDetectionRunning(false);
        
        if (frameIntervalRef.current) {
          clearInterval(frameIntervalRef.current);
          frameIntervalRef.current = null;
        }
      } else {
        alert('Error stopping detection: ' + response.data.message);
      }
    } catch (error) {
      console.error('Error stopping detection:', error);
      alert('Error stopping detection: ' + error.message);
    }
  };
  
  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
    };
  }, []);
  
  // Function to simulate movement along the path
  const simulateMovement = async () => {
    if (path.length < 2) {
      alert('Please find a path first');
      return;
    }
    
    // Initialize the simulation at the first point
    let currentIndex = 0;
    
    const interval = setInterval(async () => {
      // Move to next point
      currentIndex++;
      
      // If we reached the end, stop the simulation
      if (currentIndex >= path.length) {
        clearInterval(interval);
        return;
      }
      
      // Update current location
      const newLocation = path[currentIndex];
      setCurrentLocation(newLocation);
      
      // Update the detection service with the new location
      try {
        await axios.post(`${DETECTION_URL}/location`, {
          lat: newLocation[0],
          lon: newLocation[1]
        });
      } catch (error) {
        console.error('Error updating location:', error);
      }
    }, 1000); // Move every 1 second
  };
  
  return (
    <div className="app">
      <div className="controls">
        <h2>Navigation Controls</h2>
        <div className="input-group">
          <label>Start Location</label>
          <select 
            value={startLocation} 
            onChange={handleStartLocationChange}
          >
            {AMRITA_LOCATIONS.map((loc) => (
              <option key={`start-${loc.name}`} value={loc.name}>
                {loc.name}
              </option>
            ))}
          </select>
          <div className="coordinates">
            <input 
              type="text" 
              placeholder="Latitude" 
              value={startLat} 
              onChange={(e) => setStartLat(e.target.value)} 
            />
            <input 
              type="text" 
              placeholder="Longitude" 
              value={startLon} 
              onChange={(e) => setStartLon(e.target.value)} 
            />
          </div>
        </div>
        
        <div className="input-group">
          <label>Destination</label>
          <select 
            value={endLocation} 
            onChange={handleEndLocationChange}
          >
            {AMRITA_LOCATIONS.map((loc) => (
              <option key={`end-${loc.name}`} value={loc.name}>
                {loc.name}
              </option>
            ))}
          </select>
          <div className="coordinates">
            <input 
              type="text" 
              placeholder="Latitude" 
              value={endLat} 
              onChange={(e) => setEndLat(e.target.value)} 
            />
            <input 
              type="text" 
              placeholder="Longitude" 
              value={endLon} 
              onChange={(e) => setEndLon(e.target.value)} 
            />
          </div>
        </div>
        
        <div className="buttons">
          <button onClick={findPath}>Find Path</button>
          <button onClick={simulateMovement}>Simulate Movement</button>
          {!detectionRunning ? (
            <button onClick={startDetection}>Start Detection</button>
          ) : (
            <button onClick={stopDetection}>Stop Detection</button>
          )}
        </div>
      </div>
      
      <MapContainer 
        center={DEFAULT_CENTER} 
        zoom={17} 
        maxZoom={19}
        minZoom={15}
        maxBounds={[
          [10.89, 76.89], // Southwest corner
          [10.91, 76.91]  // Northeast corner
        ]}
        maxBoundsViscosity={1.0}
        style={{ height: '100vh', width: '100vw' }}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />
        
        {/* Show path if available */}
        {path.length > 0 && (
          <Polyline positions={path} color="blue" weight={4} />
        )}
        
        {/* Show current location */}
        <Marker position={currentLocation} />
        
        {/* Update map view when path changes */}
        <MapUpdater path={path} />
      </MapContainer>
      
      {/* Video feed */}
      <div className="video-feed">
        <img ref={videoRef} alt="Video feed" />
      </div>
      
      {/* Object detection status */}
      <div className="detection-status">
        <h3>Detected Objects: {objects.length}</h3>
        <ul>
          {objects.map((obj, index) => (
            <li key={index}>
              Class: {obj.class}, Confidence: {obj.confidence.toFixed(2)}
              {obj.is_obstacle && <strong> (Obstacle)</strong>}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

export default App; 