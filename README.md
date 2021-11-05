[Original Repo](https://github.com/microsoft/Deep3DFaceReconstruction) 

```
face_encoder = tf.keras.models.load_model("./face_encoder") # download from release and unzip
face_encoder_input = tf.keras.layers.Input((256, 256, 3))
x = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2)))(face_encoder_input)
x = face_encoder(x)
face_encoder = tf.keras.Model(face_encoder_input, x)

image = np.array(Image.open("..."))[np.newaxis,...].astype(np.float32) / 255.
coeff = face_encoder(image)
```
