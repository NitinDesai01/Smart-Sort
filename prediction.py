import warnings
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

class_labels=['battery', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 'metal', 'plastic', 'shoes', 'white-glass']

model=load_model('models/DenseNetLSTM_model.h5',compile=False)
model.compile(optimizer=Adam(learning_rate=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])


def predict(img_path):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    model_pred = model.predict(image, verbose=0)
    class_name = np.argmax(model_pred)
    class_label = class_labels[class_name]
    return class_label
