import tensorflow as tf

eye_model = tf.keras.models.load_model("eye_state_cnn.keras")
drowsy_model = tf.keras.models.load_model("drowsiness_cnn.keras")

print("Eye model:", eye_model.input_shape)
print("Drowsiness model:", drowsy_model.input_shape)

print("Both models loaded successfully!")