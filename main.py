import cv2
import torch
from flask import Flask, request, render_template, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torchvision import transforms
from datetime import datetime
from yolov5 import YOLOv5

app = Flask(__name__)
engine = create_engine('sqlite:///parking_data.db')
Base = declarative_base()

class CarEntry(Base):
    __tablename__ = 'car_entries'
    id = Column(Integer, primary_key=True)
    plate_number = Column(String)
    car_make = Column(String)
    entry_time = Column(DateTime, default=datetime.utcnow)
    exit_time = Column(DateTime)

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

yolo_model = YOLOv5("yolov5s.pt")
ocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
ocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
car_make_model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
car_make_model.eval()
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

@app.route('/upload', methods=['POST'])
def upload_image():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    yolo_results = yolo_model.predict(image)
    plate_coords = next((r['box'] for r in yolo_results if r['label'] == 'license plate'), None)
    car_coords = next((r['box'] for r in yolo_results if r['label'] == 'car'), None)
    
    if plate_coords:
        plate_img = image[int(plate_coords[1]):int(plate_coords[3]), int(plate_coords[0]):int(plate_coords[2])]
        plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
        pixel_values = ocr_processor(images=plate_img_rgb, return_tensors="pt").pixel_values
        plate_text = ocr_model.generate(pixel_values)
        plate_number = ocr_processor.batch_decode(plate_text, skip_special_tokens=True)[0]
    else:
        plate_number = "Not detected"

    if car_coords:
        car_img = image[int(car_coords[1]):int(car_coords[3]), int(car_coords[0]):int(car_coords[2])]
        car_img_tensor = transform(car_img).unsqueeze(0)
        with torch.no_grad():
            car_make_preds = car_make_model(car_img_tensor)
        car_make = car_make_preds.argmax().item()
    else:
        car_make = "Unknown"
    
    session = Session()
    car_entry = CarEntry(plate_number=plate_number, car_make=car_make)
    session.add(car_entry)
    session.commit()
    
    return jsonify({'plate_number': plate_number, 'car_make': car_make})

@app.route('/count_cars', methods=['GET'])
def count_cars():
    session = Session()
    entry_count = session.query(CarEntry).filter(CarEntry.exit_time.is_(None)).count()
    return jsonify({'active_cars': entry_count})

@app.route('/update_exit', methods=['POST'])
def update_exit():
    plate_number = request.json.get('plate_number')
    session = Session()
    car_entry = session.query(CarEntry).filter_by(plate_number=plate_number, exit_time=None).first()
    if car_entry:
        car_entry.exit_time = datetime.utcnow()
        session.commit()
        return jsonify({'status': 'exit recorded'})
    return jsonify({'status': 'plate not found'})

if __name__ == '__main__':
    app.run(debug=True)
