# Image Extraction Script

import pandas as pd
import kagglehub
import os 
import warnings
warnings.filterwarnings("ignore")


path = kagglehub.dataset_download("nih-chest-xrays/data")

# locates the parent folder of each image and adds it to the dataset as a column
def find_img_folder(dataset):
    targeted_imgs = dataset["image_index"]

    #  build a map of image filename 
    image_to_folder = {}

    for folder in os.listdir(path):
        if folder.startswith("images"):
            folder_path = os.path.join(path, folder, "images")
            try:
                for img_file in os.listdir(folder_path):
                    image_to_folder[img_file] = folder  
            except FileNotFoundError:
                continue  

    # use the map to find which folder each targeted image is in
    parent_folders = []
    for img in targeted_imgs:
        folder = image_to_folder.get(img)
        if folder:
            parent_folders.append(folder)
        else:
            print(f"Not found: {img}")

    # add extracted parent folder to dataframe as a new column
    parent_folders_df = pd.DataFrame({
        "image_index": targeted_imgs,
        "src_folder": parent_folders
        })
    dataset = pd.merge(dataset, parent_folders_df, on="image_index")

    return dataset

# Locate images and stores it in a list
def locate_imgs(dataset, loc):

    # locates the source path for all images in the dataset
    for row in range(dataset.shape[0]):
        source_path = os.path.join(path, dataset.iloc[row]["src_folder"], "images", dataset.iloc[row]["image_index"])
        loc.append(source_path)