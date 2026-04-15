import tensorflow as tf
import numpy as np
import cv2

def get_gradcam_unet(model, image, layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        
        # Focus on predicted mask (important)
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)

    # Compute weights
    weights = tf.reduce_mean(grads, axis=(1, 2))

    cam = tf.reduce_sum(
        tf.multiply(weights[:, tf.newaxis, tf.newaxis, :], conv_outputs),
        axis=-1
    )

    cam = cam.numpy()[0]

    # ReLU + normalize
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)

    return cam
def overlay_gradcam(image, cam):
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
img = cv2.imread(r"D://test\1_1.jpg")

img = cv2.resize(img, (256, 256))

input_img = np.expand_dims(img / 255.0, axis=0)
model = tf.keras.models.load_model("leaf_segmentation_model_2.h5")
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output.shape)
layer_name = "your_encoder_layer"  # VERY IMPORTANT

model = tf.keras.models.load_model("leaf_segmentation_model_2.h5")

cam = get_gradcam_unet(model, input_img, layer_name)
result = overlay_gradcam(img, cam)