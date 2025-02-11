# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Custom formatting functions for Electricity dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import libs.utils as utils
import pandas as pd
import sklearn.preprocessing
from collections import Counter
GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class StreamflowFormatter(GenericDataFormatter):
  """Defines and formats data for the electricity dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('GAGEID', DataTypes.REAL_VALUED, InputTypes.ID),
      ('time_idx', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('Q1', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('P1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('T1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('TWSA1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
      ('median_precep', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('mean_Temp', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
      ('median_allTWSA_range_yr', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
  ]

  def __init__(self):
    """Initialises formatter."""

    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=149, test_boundary=150):
    """Splits data frame into training-validation-test data frames.

    This also calibrates scaling object, and transforms data for each split.

    Args:
      df: Source data frame to split.
      valid_boundary: Starting year for validation data
      test_boundary: Starting year for test data

    Returns:
      Tuple of transformed (train, valid, test) data.
    """

    print('Formatting train-valid-test splits.')
    # df = df.loc[df.weights==0]
    encoder_len = self.get_fixed_params()['num_encoder_steps']
    index = df['time_idx']
    weights = df['weights']
    self.valid_boundary = valid_boundary
    train = df.loc[(index < valid_boundary) & (df['weights']==0)]
    valid = df.loc[(index >= valid_boundary-encoder_len) & (index < test_boundary) & (df['weights']==0)]
    test = df.loc[(index >= test_boundary-encoder_len)& (df['weights']==0)   ]
    # all_data = df

    self.set_scalers(train)

    return (self.transform_inputs(data) for data in [train, valid, test])

  def set_scalers(self, df):
    """Calibrates scalers using the data supplied.

    Args:
      df: Data to use to calibrate scalers.
    """
    print('Setting scalers with training data...')

    column_definitions = self.get_column_definition()
    id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

    # Format real scalers
    real_inputs = utils.extract_cols_from_data_type( DataTypes.REAL_VALUED, column_definitions,{InputTypes.ID, InputTypes.TIME})

    # Initialise scaler caches
    self._real_scalers = {}
    self._target_scaler = {}
    identifiers = []
    for identifier, sliced in df.groupby(id_column):

      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs].values
        targets = sliced[[target_column]].values
        self._real_scalers[identifier]  = sklearn.preprocessing.Normalizer().fit(data)

        self._target_scaler[identifier] = sklearn.preprocessing.Normalizer().fit(targets)
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].apply(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
      num_classes.append(srs.nunique())

    # Set categorical scaler outputs
    self._cat_scalers = categorical_scalers
    self._num_classes_per_cat_input = num_classes

    # Extract identifiers in case required
    self.identifiers = identifiers

  def transform_inputs(self, df):
    """Performs feature transformations.

    This includes both feature engineering, preprocessing and normalisation.

    Args:
      df: Data frame to transform.

    Returns:
      Transformed data frame.

    """

    if self._real_scalers is None and self._cat_scalers is None:
      raise ValueError('Scalers have not been set!')

    # Extract relevant columns
    column_definitions = self.get_column_definition()
    id_col = utils.get_single_col_by_input_type(InputTypes.ID,column_definitions)
    target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)
    
    real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definitions,{InputTypes.ID, InputTypes.TIME})
    categorical_inputs = utils.extract_cols_from_data_type( DataTypes.CATEGORICAL, column_definitions,{InputTypes.ID, InputTypes.TIME})

    # Transform real inputs per entity
    df_list = []

    for identifier, sliced in df.groupby(id_col):
      # print(sliced)
      if identifier in self._real_scalers:
          # Filter out any trajectories that are too short
          if len(sliced) >= self._time_steps:            
            sliced_copy = sliced.copy()
            # print(sliced_copy[real_inputs].values.shape)
            # sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)
            df_list.append(sliced_copy)
          # else:
          #     print(identifier,len(sliced),'missing timestamps')
      else:
          # print('Gauge missing during traning')
          sliced_copy = sliced.copy()
          df_gage = df[(df[id_col]==identifier)& (df['time_idx']< self.valid_boundary)]
          data = df_gage[real_inputs].values
          targets = sliced[[target_column]].values
          self._real_scalers[identifier] = sklearn.preprocessing.normalize().fit(data)
          self._target_scaler[identifier] = sklearn.preprocessing.StandardScaler().fit(targets)
          if len(sliced) >= self._time_steps:
               # print(df_gage[real_inputs].values.shape,sliced_copy.shape)
              # sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)
              df_list.append(sliced_copy)
              
    output = pd.concat(df_list, axis=0)    
    
    # print(output.shape,df.shape,'before cat')
    
    # # Format categorical inputs
    # for col in categorical_inputs:
    #   string_df = df[col].apply(str)
    #   print(output.shape,string_df.shape,col)
    #   output[col] = self._cat_scalers[col].transform(string_df)
      
    # my addition
    
    for col in categorical_inputs:
        df[col] = df[col].apply(str)
        df_list = []
        for identifier, sliced in df.groupby(id_col):            
            if len(sliced) >= self._time_steps:
                df_list.extend(self._cat_scalers[col].transform(df.loc[df[id_col]==identifier,col].values))
  
        output[col] = df_list
    return output

  def format_predictions(self, predictions):
    """Reverts any normalisation to give predictions in original scale.

    Args:
      predictions: Dataframe of model predictions.

    Returns:
      Data frame of unnormalised predictions.
    """

    if self._target_scaler is None:
      raise ValueError('Scalers have not been set!')

    column_names = predictions.columns

    df_list = []
    for identifier, sliced in predictions.groupby('identifier'):
      sliced_copy = sliced.copy()
      target_scaler = self._target_scaler[identifier]

      # for col in column_names:
      #   if col not in {'forecast_time', 'identifier'}:
      #     sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[col].values.reshape(-1,1))
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps': 24,
        'num_encoder_steps': 18,
        'num_epochs': 100,
        'early_stopping_patience': 5,
        'multiprocessing_workers': 5
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.1,
        'hidden_layer_size': 50,
        'learning_rate': 0.001,
        'minibatch_size': 64,
        'max_gradient_norm': 0.01,
        'num_heads': 4,
        'stack_size': 1
    }

    return model_params

  def get_num_samples_for_calibration(self):
    """Gets the default number of training and validation samples.

    Use to sub-sample the data for network calibration and a value of -1 uses
    all available samples.

    Returns:
      Tuple of (training samples, validation samples)
    """
    return 450000, 50000
