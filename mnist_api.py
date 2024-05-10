import uvicorn
from fastapi import FastAPI, File, UploadFile, Request
from keras.models import load_model
import numpy as np
from PIL import Image
import io

import time

from prometheus_client import Summary, start_http_server, Counter, Gauge
from prometheus_client import disable_created_metrics

# disable _created metric.
#disable_created_metrics()

REQUEST_DURATION = Summary('api_timing', 'Request duration in seconds')
counter = Counter('api_call_counter', 'number of times that API is called', ['endpoint', 'client'])
gauge = Gauge('api_runtime_secs', 'runtime of the method in seconds', ['endpoint', 'client']) 

app = FastAPI()

def format_image(img):
    
    img = img.convert('L')  # Convert to grayscale

    img = img.resize((28, 28))  # Resize the image to 28x28

    # Convert image to numpy array
    img_array = np.array(img)

    # Reshape the data_point to match the input shape of the model
    img_array = img_array.reshape(1,784)

    return img_array

def Load_Model(path: str):
    """
    Load the Keras Sequential model from the given path.
    """
    model = load_model(path)
    return model

def predict_digit(model, data_point):
    """
    Predict the digit using the loaded model.
    """
    # Normalize the data
    data_point = data_point / 255.0
    # Perform prediction
    prediction = model.predict(data_point)
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)
    return str(predicted_digit)

model_path = None

@REQUEST_DURATION.time()
@app.post('/predict')
async def predict(request:Request, upload_file: UploadFile = File(...)):
    """
    Predict the digit from the uploaded image file.
    """
    counter.labels(endpoint='/predict', client=request.client.host).inc()
    
    start = time.time()
    # Check if model is loaded
    if model_path is None:
        return {"error": "Model not loaded. Please load the model first."}
    
    # Read the uploaded image file
    contents = await upload_file.read()
    
    img = Image.open(io.BytesIO(contents))         # Load the image 
    
    img_array = format_image(img)                  # To convert into greyscale and to serialize the image array into 784 elements
    
    # Call predict_digit function to get the predicted digit
    digit = predict_digit(Load_Model(model_path), img_array)
    time_taken = time.time() - start

    gauge.labels(endpoint='/predict', client=request.client.host).set(time_taken)
    return {"Digit" : digit}
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python app.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    # start the exporter metrics service
    start_http_server(18000)
    uvicorn.run(app)