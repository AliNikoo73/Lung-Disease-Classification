import os
import shutil
import kagglehub
from sklearn.model_selection import train_test_split
import glob

def download_and_organize_dataset():
    # Download the dataset
    dataset_path = kagglehub.dataset_download('omkarmanohardalvi/lungs-disease-dataset-4-types')
    print(f"Dataset downloaded to: {dataset_path}")
    
    # Create directory structure
    base_dir = 'data'
    splits = ['train', 'val', 'test']
    classes = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']
    
    # Create directories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_dir, split, class_name), exist_ok=True)
    
    # Source directory containing the downloaded dataset
    source_dir = os.path.join(dataset_path, 'Lung Disease Dataset')
    
    # Process each class
    for class_name in classes:
        # Get all images for this class
        class_images = []
        for ext in ['*.jpeg', '*.jpg', '*.png']:
            class_images.extend(glob.glob(os.path.join(source_dir, '**', class_name, ext), recursive=True))
        
        if not class_images:
            print(f"No images found for class: {class_name}")
            continue
            
        # Split into train, validation, and test sets
        train_images, temp_images = train_test_split(class_images, test_size=0.3, random_state=42)
        val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)
        
        # Copy files to their respective directories
        def copy_files(file_list, split):
            for img_path in file_list:
                filename = os.path.basename(img_path)
                dest_path = os.path.join(base_dir, split, class_name, filename)
                shutil.copy2(img_path, dest_path)
        
        # Copy files to their respective directories
        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        copy_files(test_images, 'test')
        
        print(f"Processed {class_name}:")
        print(f"  - Training images: {len(train_images)}")
        print(f"  - Validation images: {len(val_images)}")
        print(f"  - Test images: {len(test_images)}")

if __name__ == "__main__":
    download_and_organize_dataset() 