# [Tampered Text in Images (TTI) Tianchi Competition](https://tianchi.aliyun.com/competition/entrance/532048/information?lang=en-us)

## Introduction

This repository contains the code I used for the Tampered Text in Images (TTI) Tianchi Competition. The goal of the competition is to develop models that can accurately detect and locate tampered text in images. 

## Dataset and Preprocessing

The TTI dataset contains 19,000 text images, with 15,994 of them manipulated using various techniques. Each image is annotated with a binary mask indicating the tampered location. To preprocess the data, I resized the images to 256x256 pixels and normalized the pixel values to between 0 and 1.

## Approach

For this competition, I experimented with various deep learning models and ultimately settled on using the EfficientNet-B4 architecture. EfficientNet-B4 is a powerful convolutional neural network that has achieved state-of-the-art results on many computer vision tasks.

To further improve the performance of the model, I employed transfer learning techniques by fine-tuning the pre-trained weights of the model on the TTI dataset. Additionally, I used data augmentation techniques such as random rotations, flips, and zooms to increase the variability of the training data and help prevent overfitting.


## Results

My model achieved a score of 81.18 on the competition leaderboard, which was in the 68/1256 of all submissions.

