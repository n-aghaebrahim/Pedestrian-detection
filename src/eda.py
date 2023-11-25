import yaml
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

def eda(yaml_file_path: str=None,
        data_dirs: str=None,
        hist_output: str=None,
        size_output: str=None,
        ):
    # Load YAML file
    ##yaml_file_path = "/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/data.yaml"
    ##data_dirs = "/Users/navid/work_station/github/Pedestrian-detection/data/pedestrian/"
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Extract relevant information
    train_path = data['train']
    val_path = data['val']
    test_path = data['test']
    nc = data['nc']
    class_names = data['names']

    # Create lists of images and labels for each set
    train_images = [os.path.join(data_dirs+train_path[3:], image) for image in os.listdir(data_dirs+train_path[3:])]
    val_images = [os.path.join(data_dirs+val_path[3:], image) for image in os.listdir(data_dirs+val_path[3:])]
    test_images = [os.path.join(data_dirs+test_path[3:], image) for image in os.listdir(data_dirs+test_path[3:])]

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Image': train_images + val_images + test_images,
                       'Label': ['Train'] * len(train_images) + ['Validation'] * len(val_images) + ['Test'] * len(test_images)})

    # Display basic information about the data
    print("Number of unique labels:", len(class_names))
    print("Class Names:", class_names)
    print("\nSample of the DataFrame:")
    print(df.head())




    # Save the histograms as images
    output_histograms_directory = hist_output #'../output/data_eda/'
    os.makedirs(output_histograms_directory, exist_ok=True)

    # Visualize distribution of labels
    plt.figure(figsize=(10, 6))
    df['Label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_histograms_directory, 'label_distribution.png'))
    plt.close()




    # Calculate and display number of train/validation/test images
    print("\nNumber of Train Images:", len(train_images))
    print("Number of Validation Images:", len(val_images))
    print("Number of Test Images:", len(test_images))

    # Image siz e analysis
    image_sizes = []
    for image_path in df['Image']:
        if image_path.endswith(('jpg', 'png')):
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            image_sizes.append((height, width))

    # Create a DataFrame for image sizes
    image_sizes_df = pd.DataFrame(image_sizes, columns=['Height', 'Width'])


    # Plot histograms of image heights and widths
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(image_sizes_df['Height'], bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Image Heights')
    plt.xlabel('Height')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_histograms_directory, 'image_height_distribution.png'))
    plt.close()

    plt.subplot(1, 2, 2)
    sns.histplot(image_sizes_df['Width'], bins=50, kde=True, color='salmon')
    plt.title('Distribution of Image Widths')
    plt.xlabel('Width')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_histograms_directory, 'image_width_distribution.png'))
    plt.close()

    # Save average image sizes as text files
    output_sizes_directory = size_output #'../output/data_eda/'
    os.makedirs(output_sizes_directory, exist_ok=True)

    # Calculate average image size for each label
    avg_size_by_label = image_sizes_df.groupby(df['Label']).mean()

    # Display average image size for each label
    print("\nAverage Image Size for Each Label:")
    print(avg_size_by_label)


    # Save average image sizes to text files
    for label, size in avg_size_by_label.iterrows():
        with open(os.path.join(output_sizes_directory, f'average_size_{label}.txt'), 'w') as file:
            file.write(f'Average Height: {size["Height"]}\n')
            file.write(f'Average Width: {size["Width"]}\n')
