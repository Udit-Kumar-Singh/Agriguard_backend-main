import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load the saved model and class labels
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_labels.txt', 'r') as f:
    class_labels = f.read().splitlines()

# Load and preprocess the test image
test_image = image.load_img('im_for_testing_purpose/c.commonrust.jpg', target_size=(128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image /= 255.0  # Rescale the pixel values

# Make predictions
predictions = model.predict(test_image)
predicted_class = class_labels[np.argmax(predictions)]

print("Predicted Class:", predicted_class)
print("Predicted Probabilities:", predictions)
