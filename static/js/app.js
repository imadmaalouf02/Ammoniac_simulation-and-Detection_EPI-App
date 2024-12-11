// Connect to the WebSocket server using Socket.IO
const socket = io();

// Coordonnées géographiques de l'ENSAM Meknès pour délimiter la zone d'étude
const lat_ensam = 33.85875223185346;
const lon_ensam = -5.57215015488876;

// Définir les limites de la zone d'étude (par exemple, une zone de 0.01 degré autour de l'ENSAM)
const lat_min = lat_ensam - 0.005;
const lat_max = lat_ensam + 0.005;
const lon_min = lon_ensam - 0.005;
const lon_max = lon_ensam + 0.005;

// Convertir les coordonnées géographiques en indices de grille
function geoToGrid(lat, lon, gridSize) {
    const lat_step = (lat_max - lat_min) / gridSize;
    const lon_step = (lon_max - lon_min) / gridSize;
    
    const x = Math.floor((lon - lon_min) / lon_step);
    const y = Math.floor((lat - lat_min) / lat_step);
    
    return { x, y };
}

// Initialize the map using Plotly.js
function initializeMap(gridData) {
    const gridSize = gridData.length;
    const x = [...Array(gridSize).keys()]; // X-axis indices
    const y = [...Array(gridSize).keys()]; // Y-axis indices
    const z = gridData; // The concentration data for the grid

    // Heatmap configuration
    const data = [{
        z: z,
        x: x,
        y: y,
        type: 'heatmap',
        colorscale: 'Viridis' // Color scheme for the heatmap
    }];

    const layout = {
        title: 'Ammonia Propagation',
        xaxis: { title: 'X (meters)' },
        yaxis: { title: 'Y (meters)' },
        margin: { t: 40, l: 50, r: 20, b: 50 }
    };

    // Plot the heatmap
    Plotly.newPlot('propagation-view', data, layout);
}

// Update the map dynamically as data is received
socket.on('simulation_data', (data) => {
    const gridData = data.grid_data;

    // Update the heatmap with the new data
    Plotly.react('propagation-view', [{
        z: gridData,
        type: 'heatmap',
        colorscale: 'Viridis'
    }], {
        title: 'Ammonia Propagation',
        xaxis: { title: 'X (meters)' },
        yaxis: { title: 'Y (meters)' },
        margin: { t: 40, l: 50, r: 20, b: 50 }
    });
});

// Display initial connection status
socket.on('connect', () => {
    console.log('Connected to server.');
});
