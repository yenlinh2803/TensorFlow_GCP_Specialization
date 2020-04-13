#!/usr/bin/env python
# coding: utf-8

# <h1> Feature Engineering </h1>
# 
# In this notebook, you will learn how to incorporate feature engineering into your pipeline.
# <ul>
# <li> Working with feature columns </li>
# <li> Adding feature crosses in TensorFlow </li>
# <li> Reading data from BigQuery </li>
# <li> Creating datasets using Dataflow </li>
# <li> Using a wide-and-deep model </li>
# </ul>

# **Note:**  You may ignore specific errors related to "papermill", "google-cloud-storage", and "datalab".  You may also ignore warnings related to '/home/jupyter/.local/bin'. These components and issues do not impact your ability to complete the lab.

# In[1]:


get_ipython().system('pip install --user apache-beam[gcp]==2.16.0 ')
get_ipython().system('pip install --user httplib2==0.12.0 ')


# **After doing a pip install, restart your kernel by selecting kernel from the menu and clicking Restart Kernel before proceeding further**

# In[1]:


import tensorflow as tf
import apache_beam as beam
import shutil
print(tf.__version__)


# <h2> 1. Environment variables for project and bucket </h2>
# 
# <li> Your project id is the *unique* string that identifies your project (not the project name). You can find this from the GCP Console dashboard's Home page.  My dashboard reads:  <b>Project ID:</b> cloud-training-demos </li>
# <li> Cloud training often involves saving and restoring model files. Therefore, we should <b>create a single-region bucket</b>. If you don't have a bucket already, I suggest that you create one from the GCP console (because it will dynamically check whether the bucket name you want is available) </li>
# </ol>
# <b>Change the cell below</b> to reflect your Project ID and bucket name.
# 

# In[2]:


import os
PROJECT = 'qwiklabs-gcp-04-36a643bc8aff'    # CHANGE THIS
BUCKET = 'qwiklabs-gcp-04-36a643bc8aff' # REPLACE WITH YOUR BUCKET NAME. Use a regional bucket in the region you selected.
REGION = 'us-central1' # Choose an available region for Cloud AI Platform


# In[3]:


# for bash
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '1.15' 

## ensure we're using python3 env
os.environ['CLOUDSDK_PYTHON'] = 'python3'


# In[4]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION\n\n## ensure we predict locally with our current Python environment\ngcloud config set ml_engine/local_python `which python`')


# <h2> 2. Specifying query to pull the data </h2>
# 
# Let's pull out a few extra columns from the timestamp.

# In[5]:


def create_query(phase, EVERY_N):
  if EVERY_N == None:
    EVERY_N = 4 #use full dataset
    
  #select and pre-process fields
  base_query = """
SELECT
  (tolls_amount + fare_amount) AS fare_amount,
  DAYOFWEEK(pickup_datetime) AS dayofweek,
  HOUR(pickup_datetime) AS hourofday,
  pickup_longitude AS pickuplon,
  pickup_latitude AS pickuplat,
  dropoff_longitude AS dropofflon,
  dropoff_latitude AS dropofflat,
  passenger_count*1.0 AS passengers,
  CONCAT(STRING(pickup_datetime), STRING(pickup_longitude), STRING(pickup_latitude), STRING(dropoff_latitude), STRING(dropoff_longitude)) AS key
FROM
  [nyc-tlc:yellow.trips]
WHERE
  trip_distance > 0
  AND fare_amount >= 2.5
  AND pickup_longitude > -78
  AND pickup_longitude < -70
  AND dropoff_longitude > -78
  AND dropoff_longitude < -70
  AND pickup_latitude > 37
  AND pickup_latitude < 45
  AND dropoff_latitude > 37
  AND dropoff_latitude < 45
  AND passenger_count > 0
  """
  
  #add subsampling criteria by modding with hashkey
  if phase == 'train': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} < 2".format(base_query,EVERY_N)
  elif phase == 'valid': 
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 2".format(base_query,EVERY_N)
  elif phase == 'test':
    query = "{} AND ABS(HASH(pickup_datetime)) % {} == 3".format(base_query,EVERY_N)
  return query
    
print(create_query('valid', 100)) #example query using 1% of data


# Try the query above in https://bigquery.cloud.google.com/table/nyc-tlc:yellow.trips if you want to see what it does (ADD LIMIT 10 to the query!)

# <h2> 3. Preprocessing Dataflow job from BigQuery </h2>
# 
# This code reads from BigQuery and saves the data as-is on Google Cloud Storage.  We can do additional preprocessing and cleanup inside Dataflow, but then we'll have to remember to repeat that prepreprocessing during inference. It is better to use tf.transform which will do this book-keeping for you, or to do preprocessing within your TensorFlow model. We will look at this in future notebooks. For now, we are simply moving data from BigQuery to CSV using Dataflow.
# 
# While we could read from BQ directly from TensorFlow (See: https://www.tensorflow.org/api_docs/python/tf/contrib/cloud/BigQueryReader), it is quite convenient to export to CSV and do the training off CSV.  Let's use Dataflow to do this at scale.
# 
# Because we are running this on the Cloud, you should go to the GCP Console (https://console.cloud.google.com/dataflow) to look at the status of the job. It will take several minutes for the preprocessing job to launch.

# In[6]:


get_ipython().run_cell_magic('bash', '', 'if gsutil ls | grep -q gs://${BUCKET}/taxifare/ch4/taxi_preproc/; then\n  gsutil -m rm -rf gs://$BUCKET/taxifare/ch4/taxi_preproc/\nfi')


# First, let's define a function for preprocessing the data

# In[7]:


import datetime

####
# Arguments:
#   -rowdict: Dictionary. The beam bigquery reader returns a PCollection in
#     which each row is represented as a python dictionary
# Returns:
#   -rowstring: a comma separated string representation of the record with dayofweek
#     converted from int to string (e.g. 3 --> Tue)
####
def to_csv(rowdict):
  days = ['null', 'Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
  CSV_COLUMNS = 'fare_amount,dayofweek,hourofday,pickuplon,pickuplat,dropofflon,dropofflat,passengers,key'.split(',')
  rowdict['dayofweek'] = days[rowdict['dayofweek']]
  rowstring = ','.join([str(rowdict[k]) for k in CSV_COLUMNS])
  return rowstring


####
# Arguments:
#   -EVERY_N: Integer. Sample one out of every N rows from the full dataset.
#     Larger values will yield smaller sample
#   -RUNNER: 'DirectRunner' or 'DataflowRunner'. Specfy to run the pipeline
#     locally or on Google Cloud respectively. 
# Side-effects:
#   -Creates and executes dataflow pipeline. 
#     See https://beam.apache.org/documentation/programming-guide/#creating-a-pipeline
####
def preprocess(EVERY_N, RUNNER):
  job_name = 'preprocess-taxifeatures' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')
  print('Launching Dataflow job {} ... hang on'.format(job_name))
  OUTPUT_DIR = 'gs://{0}/taxifare/ch4/taxi_preproc/'.format(BUCKET)

  #dictionary of pipeline options
  options = {
    'staging_location': os.path.join(OUTPUT_DIR, 'tmp', 'staging'),
    'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
    'job_name': 'preprocess-taxifeatures' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S'),
    'project': PROJECT,
    'runner': RUNNER,
    'num_workers' : 4,
    'max_num_workers' : 5
  }
  #instantiate PipelineOptions object using options dictionary
  opts = beam.pipeline.PipelineOptions(flags=[], **options)
  #instantantiate Pipeline object using PipelineOptions
  with beam.Pipeline(options=opts) as p:
      for phase in ['train', 'valid']:
        query = create_query(phase, EVERY_N) 
        outfile = os.path.join(OUTPUT_DIR, '{}.csv'.format(phase))
        (
          p | 'read_{}'.format(phase) >> beam.io.Read(beam.io.BigQuerySource(query=query))
            | 'tocsv_{}'.format(phase) >> beam.Map(to_csv)
            | 'write_{}'.format(phase) >> beam.io.Write(beam.io.WriteToText(outfile))
        )
  print("Done")


# Now, let's run pipeline locally. This takes upto <b>5 minutes</b>.  You will see a message "Done" when it is done.

# In[8]:


preprocess(50*10000, 'DirectRunner') 


# In[9]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://$BUCKET/taxifare/ch4/taxi_preproc/')


# ## 4. Run Beam pipeline on Cloud Dataflow

# Run pipeline on cloud on a larger sample size.

# In[10]:


get_ipython().run_cell_magic('bash', '', 'if gsutil ls | grep -q gs://${BUCKET}/taxifare/ch4/taxi_preproc/; then\n  gsutil -m rm -rf gs://$BUCKET/taxifare/ch4/taxi_preproc/\nfi')


# The following step will take <b>15-20 minutes.</b> Monitor job progress on the [Cloud Console, in the Dataflow](https://console.cloud.google.com/dataflow) section

# In[11]:


preprocess(50*100, 'DataflowRunner') 


# Once the job completes, observe the files created in Google Cloud Storage

# In[12]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls -l gs://$BUCKET/taxifare/ch4/taxi_preproc/')


# In[13]:


get_ipython().run_cell_magic('bash', '', '#print first 10 lines of first shard of train.csv\ngsutil cat "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-*" | head')


# ## 5. Develop model with new inputs
# 
# Download the first shard of the preprocessed data to enable local development.

# In[14]:


get_ipython().run_cell_magic('bash', '', 'if [ -d sample ]; then\n  rm -rf sample\nfi\nmkdir sample\ngsutil cat "gs://$BUCKET/taxifare/ch4/taxi_preproc/train.csv-00000-of-*" > sample/train.csv\ngsutil cat "gs://$BUCKET/taxifare/ch4/taxi_preproc/valid.csv-00000-of-*" > sample/valid.csv')


# We have two new inputs in the INPUT_COLUMNS, three engineered features, and the estimator involves bucketization and feature crosses.

# In[15]:


get_ipython().run_cell_magic('bash', '', 'grep -A 20 "INPUT_COLUMNS =" taxifare/trainer/model.py')


# In[16]:


get_ipython().run_cell_magic('bash', '', 'grep -A 50 "build_estimator" taxifare/trainer/model.py')


# In[17]:


get_ipython().run_cell_magic('bash', '', 'grep -A 15 "add_engineered(" taxifare/trainer/model.py')


# Try out the new model on the local sample (this takes <b>5 minutes</b>) to make sure it works fine.

# In[18]:


get_ipython().run_cell_magic('bash', '', 'rm -rf taxifare.tar.gz taxi_trained\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare\npython -m trainer.task \\\n  --train_data_paths=${PWD}/sample/train.csv \\\n  --eval_data_paths=${PWD}/sample/valid.csv  \\\n  --output_dir=${PWD}/taxi_trained \\\n  --train_steps=10 \\\n  --job-dir=/tmp')


# In[19]:


get_ipython().run_cell_magic('bash', '', 'ls taxi_trained/export/exporter/')


# You can use ```saved_model_cli``` to look at the exported signature. Note that the model doesn't need any of the engineered features as inputs. It will compute latdiff, londiff, euclidean from the provided inputs, thanks to the ```add_engineered``` call in the serving_input_fn.

# In[20]:


get_ipython().run_cell_magic('bash', '', 'model_dir=$(ls ${PWD}/taxi_trained/export/exporter | tail -1)\nsaved_model_cli show --dir ${PWD}/taxi_trained/export/exporter/${model_dir} --all')


# In[21]:


get_ipython().run_cell_magic('writefile', '/tmp/test.json', '{"dayofweek": "Sun", "hourofday": 17, "pickuplon": -73.885262, "pickuplat": 40.773008, "dropofflon": -73.987232, "dropofflat": 40.732403, "passengers": 2}')


# In[22]:


get_ipython().run_cell_magic('bash', '', 'model_dir=$(ls ${PWD}/taxi_trained/export/exporter)\ngcloud ai-platform local predict \\\n  --model-dir=${PWD}/taxi_trained/export/exporter/${model_dir} \\\n  --json-instances=/tmp/test.json')


# ## 6. Train on cloud
# 
# This will take <b> 10-15 minutes </b> even though the prompt immediately returns after the job is submitted. Monitor job progress on the [Cloud Console, in the AI Platform](https://console.cloud.google.com/mlengine) section and wait for the training job to complete.
# 

# In[24]:


get_ipython().run_cell_magic('bash', '', 'OUTDIR=gs://${BUCKET}/taxifare/ch4/taxi_trained\nJOBNAME=lab4a_$(date -u +%y%m%d_%H%M%S)\necho $OUTDIR $REGION $JOBNAME\ngsutil -m rm -rf $OUTDIR\ngcloud ai-platform jobs submit training $JOBNAME \\\n  --region=$REGION \\\n  --module-name=trainer.task \\\n  --package-path=${PWD}/taxifare/trainer \\\n  --job-dir=$OUTDIR \\\n  --staging-bucket=gs://$BUCKET \\\n  --scale-tier=BASIC \\\n  --runtime-version=$TFVERSION \\\n  -- \\\n  --train_data_paths="gs://$BUCKET/taxifare/ch4/taxi_preproc/train*" \\\n  --eval_data_paths="gs://${BUCKET}/taxifare/ch4/taxi_preproc/valid*"  \\\n  --train_steps=5000 \\\n  --output_dir=$OUTDIR')


# The RMSE is now 8.33249, an improvement over the 9.3 that we were getting ... of course, we won't know until we train/validate on a larger dataset. Still, this is promising. But before we do that, let's do hyper-parameter tuning.
# 
# <b>Use the Cloud Console link to monitor the job and do NOT proceed until the job is done.</b>

# In[27]:


get_ipython().run_cell_magic('bash', '', 'gsutil ls gs://${BUCKET}/taxifare/ch4/taxi_trained/export/exporter | tail -1')


# In[28]:


get_ipython().run_cell_magic('bash', '', 'model_dir=$(gsutil ls gs://${BUCKET}/taxifare/ch4/taxi_trained/export/exporter | tail -1)\nsaved_model_cli show --dir ${model_dir} --all')


# In[29]:


get_ipython().run_cell_magic('bash', '', 'model_dir=$(gsutil ls gs://${BUCKET}/taxifare/ch4/taxi_trained/export/exporter | tail -1)\ngcloud ai-platform local predict \\\n  --model-dir=${model_dir} \\\n  --json-instances=/tmp/test.json')


# ### Optional: deploy model to cloud

# In[30]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="feateng"\nMODEL_VERSION="v1"\nMODEL_LOCATION=$(gsutil ls gs://${BUCKET}/taxifare/ch4/taxi_trained/export/exporter | tail -1)\necho "Run these commands one-by-one (the very first time, you\'ll create a model and then create a version)"\n#gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME}\n#gcloud ai-platform delete ${MODEL_NAME}\ngcloud ai-platform models create ${MODEL_NAME} --regions $REGION\ngcloud ai-platform versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version $TFVERSION')


# In[31]:


get_ipython().run_cell_magic('bash', '', 'gcloud ai-platform predict --model=feateng --version=v1 --json-instances=/tmp/test.json')


# Copyright 2016 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
