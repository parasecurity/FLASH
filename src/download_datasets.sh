#!/bin/bash
mkdir -p downloaded_datasets

echo "Attempting to download with curl..."
curl -k -L -o downloaded_datasets/Tuandromd.zip "https://www.archive.ics.uci.edu/dataset/855/tuandromd+(tezpur+university+android+malware+dataset).zip"
echo "Extracting the downloaded zip file..."
unzip -o downloaded_datasets/Tuandromd.zip -d downloaded_datasets/
echo "Deleting the Tuandromd zip file after extraction..."
rm downloaded_datasets/Tuandromd.zip
echo "Tuandromd download and extraction completed!"

echo "Downloading heart disease dataset from Kaggle..."
curl -L -o downloaded_datasets/personal-key-indicators-of-heart-disease.zip \
  https://www.kaggle.com/api/v1/datasets/download/kamilpytlak/personal-key-indicators-of-heart-disease
echo "Extracting the Kaggle dataset..."
mkdir -p heart-disease-temp
unzip -o downloaded_datasets/personal-key-indicators-of-heart-disease.zip -d heart-disease-temp
# Move contents of 2020 folder to main folder
echo "Moving contents of 2020 folder to main folder..."
mv heart-disease-temp/2020/* downloaded_datasets
echo "Cleaning up temporary files..."
rm -rf heart-disease-temp
rm personal-key-indicators-of-heart-disease.zip
