import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Load and preprocess image for prediction"""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def apply_gradcam(model, img_array, layer_name=None):
    """Apply Grad-CAM to visualize model attention"""
    if layer_name is None:
        # Find the last convolutional layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)

    cam = cv2.resize(cam.numpy(), target_size)
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    return cam

def visualize_prediction(img_path, model, class_names, save_path=None):
    """Visualize prediction with Grad-CAM"""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Get prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Apply Grad-CAM
    cam = apply_gradcam(model, img_array)

    # Create visualization
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Prediction: {class_names[predicted_class]}\nConfidence: {confidence:.2%}')
    plt.axis('off')

    # Grad-CAM visualization
    plt.subplot(1, 2, 2)
    plt.imshow(img)
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.title('Grad-CAM Visualization')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to image for prediction')
    parser.add_argument('--class_names', type=str, nargs='+', required=True, help='List of class names')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Output directory for visualizations')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model(args.model_path)

    # Get image filename for output
    image_filename = os.path.basename(args.image_path)
    output_path = os.path.join(args.output_dir, f'prediction_{image_filename}')

    # Make and visualize prediction
    visualize_prediction(
        args.image_path,
        model,
        args.class_names,
        save_path=output_path
    )

    print(f"Prediction visualization saved to {output_path}")

if __name__ == '__main__':
    main() 