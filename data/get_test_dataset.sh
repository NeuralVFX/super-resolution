#!/bin/bash

URL=http://files.fast.ai/data/imagenet-sample-train.tar.gz
TAR_FILE=./data/imagenet-sample-train.tar.gz
wget $URL -O $TAR_FILE
tar -xvzf $TAR_FILE -C ./data/