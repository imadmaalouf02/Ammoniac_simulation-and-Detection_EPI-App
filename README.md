# Ammonia Leak Simulation

## Setup Instructions

1. Create a new Anaconda environment:
```bash
conda create -n ammonia-sim python=3.9
```

2. Activate the environment:
```bash
conda activate ammonia-sim
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:3000
```

## Project Structure

- `app.py`: Main Flask application with WebSocket server
- `public/`: Static files (HTML, CSS, JavaScript)
  - `index.html`: Main webpage
  - `styles.css`: Styling
  - `app.js`: Frontend JavaScript
  - `js/`: JavaScript modules
    - `chart-config.js`: Chart configuration
    - `ui-handlers.js`: UI update functions
    - `chart-updater.js`: Chart update functions