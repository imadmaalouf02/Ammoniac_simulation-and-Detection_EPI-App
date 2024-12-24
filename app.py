from flask import Flask, render_template, request, send_file
from flask_socketio import SocketIO
import math
import time
import threading
import os
from ultralytics import YOLOv10
import supervision as sv
import cv2
import os
from flask import Response


# Créer les répertoires nécessaires si inexistants
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


app = Flask(__name__)
socketio = SocketIO(app)

# Coordonnées des capteurs
sensors = [
    {"id": 1, "lat": 33.85563147149165, "lon": -5.572377739710788, "concentration": 0, "name": "Administrateur"},
    {"id": 2, "lat": 33.85713810485788, "lon": -5.571694444520745, "concentration": 0, "name": "Département Informatique"},
    {"id": 3, "lat": 33.857751186152115, "lon": -5.5707205525246195, "concentration": 0, "name": "Département Mécanique"}
]

CONCENTRATION_INCREMENT = 1
PROPAGATION_RADIUS = 0.005
MAX_RADIUS = 0.01
ALERT_THRESHOLD = 50

model = YOLOv10('best.pt')  # Charger le modèle YOLO

def simulate_ammonia_propagation():
    max_concentration = 200
    while True:
        alerts = []
        for sensor in sensors:
            if sensor["concentration"] < max_concentration:
                sensor["concentration"] += CONCENTRATION_INCREMENT
            
            if sensor["concentration"] >= ALERT_THRESHOLD:
                alerts.append(f"Capteur {sensor['id']} a détecté une fuite d'ammoniac!")

        sensor_circles = []
        for sensor in sensors:
            radius = (sensor["concentration"] / max_concentration) * MAX_RADIUS
            sensor_circles.append({
                "lat": sensor["lat"],
                "lon": sensor["lon"],
                "radius": radius,
                "concentration": sensor["concentration"],
                "name": sensor["name"]
            })

        socketio.emit("update_data", {"sensor_circles": sensor_circles, "alerts": alerts})
        time.sleep(0.5)

# Fonction pour annoter une image
def annotate_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    output_path = "static/output/annotated_image.jpg"
    cv2.imwrite(output_path, annotated_image)
    return output_path



def generate_frames():
    cap = cv2.VideoCapture(0)  # Accéder à la caméra par défaut
    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_path = "static/output/processed_video.mp4"
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        out.write(annotated_frame)

    cap.release()
    out.release()
    return output_path






# Page d'accueil
@app.route('/')
def home():
    return render_template('home.html')

# Page de simulation
@app.route('/simulation')
def simulation():
    return render_template('index.html')

# Page de simulation
@app.route('/detection')
def detection():
    return render_template('HDet.html')

# Page de détection
@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        image = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image.filename)  # Utilisation de UPLOAD_FOLDER
        image.save(image_path)

        output_path = annotate_image(image_path)
        return render_template('Detection.html', input_image=image_path, output_image=output_path)

    return render_template('Detection.html', input_image=None, output_image=None)





# Page de détection pour les vidéos
@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        video = request.files['video']
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)  # Enregistrer la vidéo dans UPLOAD_FOLDER
        video.save(video_path)

        output_path = process_video(video_path)
        return render_template('VideoDetection.html', input_video=video_path, output_video=output_path)

    return render_template('VideoDetection.html', input_video=None, output_video=None)




# Page pour le flux en direct de la caméra
@app.route('/camera')
def camera():
    return render_template('CameraDetection.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    threading.Thread(target=simulate_ammonia_propagation).start()
    socketio.run(app, debug=True)
    #