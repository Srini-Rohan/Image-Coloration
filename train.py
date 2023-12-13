import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from dataset import Dataset
from model import Pix2PixGAN
from utils import generate_images
(color_dataset,gray_dataset),(color_dataset_test,gray_dataset_test) = Dataset('/dataset',256)

model = Pix2PixGAN(256)


model.fit(tf.data.Dataset.zip((gray_dataset, color_dataset)),epochs = 10)

for example_input, example_target in tf.data.Dataset.zip((gray_dataset_test,color_dataset_test)).take(11):
  generate_images(model.generator, example_input, example_target)