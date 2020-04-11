#!/usr/bin/env python
# coding: utf-8

# <h1> 2d. Distributed training and monitoring </h1>
# 
# In this notebook, we refactor to call ```train_and_evaluate``` instead of hand-coding our ML pipeline. This allows us to carry out evaluation as part of our training loop instead of as a separate step. It also adds in failure-handling that is necessary for distributed training capabilities.
# 
# We also use TensorBoard to monitor the training.

# In[1]:


from google.cloud import bigquery
import tensorflow as tf
import numpy as np
import shutil
print(tf.__version__)


# <h2> Input </h2>
# 
# Read data created in Lab1a, but this time make it more general, so that we are reading in batches.  Instead of using Pandas, we will use add a filename queue to the TensorFlow graph.

# In[2]:


CSV_COLUMNS = ['fare_amount', 'pickuplon','pickuplat','dropofflon','dropofflat','passengers', 'key']
LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [-74.0], [40.0], [-74.0], [40.7], [1.0], ['nokey']]

def read_dataset(filename, mode, batch_size = 512):
  def decode_csv(value_column):
    columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    label = features.pop(LABEL_COLUMN)
    return features, label

  # Create list of file names that match "glob" pattern (i.e. data_file_*.csv)
  filenames_dataset = tf.data.Dataset.list_files(filename)
  # Read lines from text files
  textlines_dataset = filenames_dataset.flat_map(tf.data.TextLineDataset)
  # Parse text lines as comma-separated values (CSV)
  dataset = textlines_dataset.map(decode_csv)

  # Note:
  # use tf.data.Dataset.flat_map to apply one to many transformations (here: filename -> text lines)
  # use tf.data.Dataset.map      to apply one to one  transformations (here: text line -> feature list)

  if mode == tf.estimator.ModeKeys.TRAIN:
      num_epochs = None # indefinitely
      dataset = dataset.shuffle(buffer_size = 10 * batch_size)
  else:
      num_epochs = 1 # end-of-input after this

  dataset = dataset.repeat(num_epochs).batch(batch_size)

  return dataset


# <h2> Create features out of input data </h2>
# 
# For now, pass these through.  (same as previous lab)

# In[3]:


INPUT_COLUMNS = [
    tf.feature_column.numeric_column('pickuplon'),
    tf.feature_column.numeric_column('pickuplat'),
    tf.feature_column.numeric_column('dropofflat'),
    tf.feature_column.numeric_column('dropofflon'),
    tf.feature_column.numeric_column('passengers'),
]

def add_more_features(feats):
  # Nothing to add (yet!)
  return feats

feature_cols = add_more_features(INPUT_COLUMNS)


# <h2> Serving input function </h2>
# Defines the expected shape of the JSON feed that the modelwill receive once deployed behind a REST API in production.

# In[4]:


## TODO: Create serving input function
# Defines the expected shape of the JSON feed that the model
# will receive once deployed behind a REST API in production.
def serving_input_fn():
    json_feature_placeholders = {
        'pickuplon' : tf.placeholder(tf.float32, [None]),
        'pickuplat' : tf.placeholder(tf.float32, [None]),
        'dropofflat' : tf.placeholder(tf.float32, [None]),
        'dropofflon' : tf.placeholder(tf.float32, [None]),
        'passengers' : tf.placeholder(tf.float32, [None]),
    }
    # You can transforma data here from the input format to the format expected by your model.
    features = json_feature_placeholders # no transformation needed
    return tf.estimator.export.ServingInputReceiver(features, json_feature_placeholders)


# <h2> tf.estimator.train_and_evaluate </h2>

# In[6]:


## TODO: Create train and evaluate function using tf.estimator
def train_and_evaluate(output_dir, num_train_steps):
    estimator = tf.estimator.LinearRegressor(
                       model_dir = output_dir,
                       feature_columns = feature_cols)
    
    train_spec=tf.estimator.TrainSpec(
                       input_fn = lambda: read_dataset('./taxi-train.csv', mode = tf.estimator.ModeKeys.TRAIN),
                       max_steps = num_train_steps)

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

    eval_spec=tf.estimator.EvalSpec(
                       input_fn = lambda: read_dataset('./taxi-valid.csv', mode = tf.estimator.ModeKeys.EVAL),
                       steps = None,
                       start_delay_secs = 1, # start evaluating after N seconds
                       throttle_secs = 10,  # evaluate every N seconds
                       exporters = exporter)
    
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# ## Monitor training with TensorBoard
# 
# To activate TensorBoard within the JupyterLab UI navigate to "<b>File</b>" - "<b>New Launcher</b>".   Then double-click the 'Tensorboard' icon on the bottom row.
# 
# TensorBoard 1 will appear in the new tab.  Navigate through the three tabs to see the active TensorBoard.   The 'Graphs' and 'Projector' tabs offer very interesting information including the ability to replay the tests.
# 
# You may close the TensorBoard tab when you are finished exploring.

# In[7]:


OUTDIR = './taxi_trained'


# <h2>Run training</h2>

# In[8]:


# Run training    
shutil.rmtree(OUTDIR, ignore_errors = True) # start fresh each time
tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
train_and_evaluate(OUTDIR, num_train_steps = 2000)


# ## Challenge Exercise
# 
# Modify your solution to the challenge exercise in c_dataset.ipynb appropriately.

# Copyright 2017 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
