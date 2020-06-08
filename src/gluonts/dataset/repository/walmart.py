# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
from pathlib import Path
import os
import json

import pandas as pd
import numpy as np

def generate_walmart_dataset(
    dataset_path: Path, pandas_freq: str, prediction_length: int
):  
    # test.csv has weekly sales removed thus we 
    # cannot use it for the test set
    #test_path = f"{dataset_path}/test.csv" 
    train_path = f"{dataset_path}/train.csv"

    if not os.path.exists(train_path):
        raise RuntimeError(
            f"Wallmart data is available on Kaggle (https://www.kaggle.com/bletchley/course-material-walmart-challenge/download). "
            f"Please supply the files at {dataset_path}."
        )
    
    train_data = pd.read_csv(train_path)

