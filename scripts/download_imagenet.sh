#!/bin/bash
curl -L -o ./data/imagenet100.zip\
  https://www.kaggle.com/api/v1/datasets/download/ambityga/imagenet100
unzip -l data/imagenet100.zip

