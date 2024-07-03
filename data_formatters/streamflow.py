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
import numpy as np
class scalar_y_class:
    def fit_transform(self,x):
        x = np.log10(x)
        return x
    def transform(self,x):
        return np.log10(x)
    def inverse_transform(self,x):
        return 10**x

class StreamflowFormatter(GenericDataFormatter):
  """Defines and formats data for the electricity dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """
      # ('P1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      # ('T1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      # ('TWSA1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      
      # ('P1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      # ('T1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      # ('TWSA1', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      
  _column_definition = [
      ('GAGEID', DataTypes.REAL_VALUED, InputTypes.ID),
      ('time_idx', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('Q_mon', DataTypes.REAL_VALUED, InputTypes.TARGET),

       ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT), 
  ]
  
  
  real_known = ['dewpoint_temperature_2m',
       'potential_evaporation_sum', 'snow_depth_water_equivalent',
       'surface_net_solar_radiation_sum',
       'surface_net_thermal_radiation_sum', 'surface_pressure',
       'temperature_2m', 'total_precipitation_sum',
       'u_component_of_wind_10m', 'v_component_of_wind_10m',
       'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
       'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
       'snowfall_sum', 'snowmelt_sum']
  
  for var_real in real_known :
      _column_definition.append((var_real, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))
      
  static_cat_inputs = ['clz_cl_smj', 'fec_cl_smj','lit_cl_smj','fmh_cl_smj','tbi_cl_smj',]
  for var_stcat in static_cat_inputs :
      _column_definition.append((var_stcat, DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT))
     
      
  static_real_inputs = ['inu_pc_smx', 'swc_pc_s07', 'glc_pc_s08', 'cmi_ix_s12',
       'pet_mm_s10', 'gwt_cm_sav', 'tmp_dc_s01', 'glc_pc_s05',
       'pnv_pc_s10', 'pet_mm_s08', 'wet_pc_s04', 'snw_pc_s12',
       'pet_mm_s07', 'snw_pc_s08', 'glc_pc_s16', 'aet_mm_syr',
       'pop_ct_usu', 'glc_pc_s17', 'hdi_ix_sav', 'pre_mm_s07',
       'tmp_dc_s05', 'wet_pc_sg1', 'pnv_pc_s11', 'glc_pc_s10',
       'pnv_pc_s04', 'tmp_dc_s11', 'snw_pc_s02', 'snw_pc_s04',
       'snw_pc_smx', 'cmi_ix_s08', 'tmp_dc_smx', 'urb_pc_sse',
       'slt_pc_sav', 'pst_pc_sse', 
       'dis_m3_pmx', 'snw_pc_s06', 'pnv_pc_s13', 'swc_pc_s05',
       'tmp_dc_smn', 'glc_pc_s06', 'pre_mm_s11', 'prm_pc_sse',
       'glc_pc_s12', 'aet_mm_s06', 'snw_pc_s07', 'wet_pc_s08',
       'ire_pc_sse', 'lkv_mc_usu', 'swc_pc_s03', 'tmp_dc_s07',
       'cmi_ix_s06', 'snw_pc_syr', 'pre_mm_s04', 'pet_mm_s05',
       'pet_mm_syr', 'tmp_dc_s04', 'snw_pc_s01', 'swc_pc_s01',
       'ari_ix_sav', 'cmi_ix_s07', 'snw_pc_s11', 'swc_pc_s12',
       'dis_m3_pmn', 'tmp_dc_s03', 'run_mm_syr', 'aet_mm_s02',
       'snw_pc_s09', 'pre_mm_s06', 'dis_m3_pyr', 'swc_pc_syr',
       'wet_pc_s05', 'glc_pc_s13', 'wet_pc_sg2', 'aet_mm_s11',
       'swc_pc_s11', 'pre_mm_s08', 'snw_pc_s05', 'nli_ix_sav',
       'tmp_dc_s06', 'soc_th_sav', 'pac_pc_sse', 'gdp_ud_ssu',
       'swc_pc_s08', 'glc_pc_s09', 'swc_pc_s02', 'pnv_pc_s06',
       'inu_pc_smn', 'snw_pc_s10', 'aet_mm_s09', 'rev_mc_usu',
       'pnv_pc_s05', 'pet_mm_s04', 'cmi_ix_s11', 'cmi_ix_s04',
       'snd_pc_sav', 'for_pc_sse', 'gla_pc_sse', 'pet_mm_s11',
       'swc_pc_s09', 'tmp_dc_s08', 'wet_pc_s02', 'gdp_ud_sav',
       'glc_pc_s14', 'pet_mm_s02', 'aet_mm_s05', 'aet_mm_s01',
       'pet_mm_s09', 'glc_pc_s01',  'slp_dg_sav',
       'aet_mm_s04', 'rdd_mk_sav', 'cly_pc_sav', 'aet_mm_s10',
       'sgr_dk_sav', 'glc_pc_s18', 'cls_cl_smj', 'glc_pc_s02',
       'tmp_dc_s02', 'pre_mm_s01', 'ero_kh_sav', 'ele_mt_smx',
       'cmi_ix_s03', 'wet_pc_s01', 'aet_mm_s08', 'wet_pc_s03',
       'wet_pc_s09', 'pre_mm_s09', 'pnv_pc_s01', 'crp_pc_sse',
       'inu_pc_slt', 'tmp_dc_syr', 'pnv_pc_s09', 
       'pre_mm_s05', 'hft_ix_s93', 'pnv_pc_s08', 'pet_mm_s12',
       'wet_pc_s07', 'glc_pc_s03', 'glc_cl_smj', 'pre_mm_s10',
        'cmi_ix_s09', 'glc_pc_s15', 'pre_mm_syr',
       'pnv_pc_s14', 'ele_mt_sav', 'ele_mt_smn', 'glc_pc_s22',
       'tec_cl_smj', 'pnv_pc_s15', 'hft_ix_s09', 'lka_pc_sse',
       'aet_mm_s07', 'pre_mm_s12', 'swc_pc_s06', 
       'pnv_pc_s07', 'pet_mm_s01', 'glc_pc_s04', 'glc_pc_s11',
       'ppd_pk_sav', 'pnv_pc_s03', 'glc_pc_s19', 'pnv_pc_s12',
       'pet_mm_s03', 'pnv_cl_smj', 'riv_tc_usu', 'dor_pc_pva',
       'kar_pc_sse', 'swc_pc_s04', 'cmi_ix_s10', 'tmp_dc_s09',
       'ria_ha_usu', 'snw_pc_s03', 'cmi_ix_s05', 'pre_mm_s03',
       'tmp_dc_s10', 'tmp_dc_s12', 'aet_mm_s12', 'cmi_ix_syr',
       'glc_pc_s07', 'glc_pc_s21', 'glc_pc_s20', 'swc_pc_s10',
       'pet_mm_s06', 'wet_pc_s06', 'pre_mm_s02', 'aet_mm_s03',
       'pnv_pc_s02', 'cmi_ix_s01', 'cmi_ix_s02']
  for var_streal in static_real_inputs :
          _column_definition.append((var_streal, DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT))

  def __init__(self):
    """Initialises formatter."""
    self.y_scalar = scalar_y_class()
    self.identifiers = None
    self._real_scalers = None
    self._cat_scalers = None
    self._target_scaler = None
    self._num_classes_per_cat_input = None
    self._time_steps = self.get_fixed_params()['total_time_steps']

  def split_data(self, df, valid_boundary=225, test_boundary=245):
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
    # weights = df['weights']
    self.valid_boundary = valid_boundary
    # df = df.loc[(df['COMID']>75000000 ) ]#|(df['COMID']<71000000 )]
    # print(df.shape,"__________________")
    # df['weights']+=1
    # df['Q1'] = np.log10(df['Q1'])
    train = df.loc[(index < valid_boundary) ]
    valid = df.loc[(index >= (valid_boundary-encoder_len)) & (index < test_boundary) ]
    test = df.loc[index >= (test_boundary-encoder_len) ]
    # all_data = df

    self.set_scalers(train)
    print(train.shape,valid.shape,test.shape)
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
    sc1 = sklearn.preprocessing.StandardScaler().fit(df[real_inputs])
    sc2  = sklearn.preprocessing.StandardScaler().fit(df[[target_column]])
    for identifier, sliced in df.groupby(id_column):
      # print(sliced.columns,sliced.head(2))  
      if len(sliced) >= self._time_steps:

        data = sliced[real_inputs]
        targets = sliced[[target_column]]
        self._real_scalers[identifier]  = sc1#sklearn.preprocessing.StandardScaler().fit(df[real_inputs].values)#data

        self._target_scaler[identifier] = sc2# sklearn.preprocessing.StandardScaler().fit(df[target_column].values.reshape(-1,1))# targets
      identifiers.append(identifier)

    # Format categorical scalers
    categorical_inputs = utils.extract_cols_from_data_type(
        DataTypes.CATEGORICAL, column_definitions,
        {InputTypes.ID, InputTypes.TIME})

    categorical_scalers = {}
    num_classes = []
    for col in categorical_inputs:
      # Set all to str so that we don't have mixed integer/string columns
      srs = df[col].astype(int).astype(str)
      categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs)
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
    print(df.shape)
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
      if identifier in self._real_scalers:
          # Filter out any trajectories that are too short
          if len(sliced) >= self._time_steps:            
            sliced_copy = sliced.copy()
            sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs])
            df_list.append(sliced_copy)
          # else:
          #      print(identifier,len(sliced),'missing timestamps')
          else:
              # print(identifier,id_col,"____________________________________________________< timestep")
              pass
           # # print('Gauge missing during traning')
            # sliced_copy = sliced.copy()
            # df_gage = df[(df[id_col]==identifier)]
            # data = df_gage[real_inputs].values
            # targets = sliced[[target_column]].values
            # self._real_scalers[identifier] = self._real_scalers[list(self._real_scalers.keys())[0]]
            # self._target_scaler[identifier] = self._target_scalers[list(self._target_scalers.keys())[0]]
            # if len(sliced) >= self._time_steps:
            #      # print(df_gage[real_inputs].values.shape,sliced_copy.shape)
            #      sliced_copy[real_inputs] = self._real_scalers[identifier].transform(sliced_copy[real_inputs].values)
            #      df_list.append(sliced_copy)
              
    output = pd.concat(df_list, axis=0)    
    
    # print(output.shape,df.shape,'before cat')
    
    # # Format categorical inputs
    # for col in categorical_inputs:
      # string_df = df[col].apply(int).apply(str)
      # print(output.shape,string_df.shape,col)
      
      # output[col] = self._cat_scalers[col].transform(string_df)
    
    for col in categorical_inputs:
        
        src = df[col].apply(str)
        df_list = []
        # print(col,self._cat_scalers[col].classes_ )
        for identifier, sliced in df.groupby(id_col): 
            if identifier in self._real_scalers:
                if len(sliced) >= self._time_steps:
                    sliced_copy = sliced.copy()
                    df_list.extend(self._cat_scalers[col].transform(sliced_copy[col].astype('str')))
                else:
                    pass
                    # print(identifier,col,"____________________________________________________< timestep")
        # print(output.shape,len(df_list),col)
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

      for col in column_names:
         if col not in {'forecast_time', 'identifier'}:
           sliced_copy[col] = target_scaler.inverse_transform(sliced_copy[[col]])
      df_list.append(sliced_copy)

    output = pd.concat(df_list, axis=0)

    return output

  # Default params
  def get_fixed_params(self):
    """Returns fixed model parameters for experiments."""

    fixed_params = {
        'total_time_steps':15,
        'num_encoder_steps': 12,
        'num_epochs': 500,
        'early_stopping_patience': 7,
        'multiprocessing_workers': 19
    }

    return fixed_params

  def get_default_model_params(self):
    """Returns default optimised model parameters."""

    model_params = {
        'dropout_rate': 0.27,
        'hidden_layer_size': 80,
        'learning_rate': 0.000005,
        'minibatch_size': 1024,
        'max_gradient_norm': 75,
        'num_heads': 16,
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
    return 340000,33000