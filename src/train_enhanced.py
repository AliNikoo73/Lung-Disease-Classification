import os
import argparse
from enhanced_model import EnhancedLungDiseaseModel
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import ModelCheckpoint
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train enhanced lung disease classification model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--model_name', type=str, default='EfficientNetV2B0', help='Base model name')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Define class names
    class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'NORMAL', 'Tuberculosis', 'Viral Pneumonia']

    # Initialize model
    model_handler = EnhancedLungDiseaseModel(
        img_size=(args.img_size, args.img_size),
        num_classes=len(class_names)
    )

    # Create data generators
    datagen = model_handler.create_enhanced_datagen()
    
    train_generator = datagen.flow_from_directory(
        os.path.join(args.data_dir, 'train'),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        classes=class_names
    )

    validation_generator = datagen.flow_from_directory(
        os.path.join(args.data_dir, 'val'),  # Changed from 'train' to 'val'
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        classes=class_names
    )

    test_generator = datagen.flow_from_directory(
        os.path.join(args.data_dir, 'test'),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        classes=class_names
    )

    # Create model with attention
    model, base_model = model_handler.create_model_with_attention(args.model_name)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Get callbacks
    callbacks = model_handler.get_callbacks(
        os.path.join(args.output_dir, f'{args.model_name}_best.h5')
    )

    # Phase 1: Train with frozen base model
    print("Phase 1: Training with frozen base model...")
    base_model.trainable = False
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs // 2,
        callbacks=callbacks
    )

    # Phase 2: Fine-tune with unfrozen base model
    print("Phase 2: Fine-tuning with unfrozen base model...")
    base_model.trainable = True
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs // 2,
        callbacks=callbacks
    )

    # Evaluate model
    print("Evaluating model...")
    metrics = model_handler.evaluate_model(model, test_generator)
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        f.write("\n\nROC AUC Score:\n")
        f.write(str(metrics['roc_auc']))

    # Visualize results
    print("Generating visualizations...")
    model_handler.visualize_results(
        model,
        test_generator,
        args.output_dir
    )

    # Save final model
    model.save(os.path.join(args.output_dir, f'{args.model_name}_final.h5'))
    print(f"Training completed. Results saved in {args.output_dir}")

def create_enhanced_datagen():
    return ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

def train_with_mixup(model, train_generator, validation_generator, epochs):
    # Calculate class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    
    # Phase 1: Train with frozen base model
    print("Phase 1: Training with frozen base model...")
    model.layers[0].trainable = False
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0005)  # Reduced learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
    
    history1 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs//2,
        callbacks=[
            early_stopping,
            reduce_lr,
            ModelCheckpoint(
                'output/EfficientNetV2B0_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ],
        class_weight=class_weight_dict
    )
    
    # Phase 2: Fine-tune with unfrozen base model
    print("Phase 2: Fine-tuning with unfrozen base model...")
    model.layers[0].trainable = True
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001)  # Further reduced learning rate
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs//2,
        callbacks=[
            early_stopping,
            reduce_lr,
            ModelCheckpoint(
                'output/EfficientNetV2B0_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ],
        class_weight=class_weight_dict
    )
    
    return {**history1.history, **history2.history}

if __name__ == '__main__':
    main() 