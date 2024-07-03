import argparse
import datetime as dte
import os

import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf



def main(expt_name, use_gpu, restart_opt, model_folder, hyperparam_iterations,data_csv_path, data_formatter):
  """Runs main hyperparameter optimization routine.

  Args:
    expt_name: Name of experiment
    use_gpu: Whether to run tensorflow with GPU operations
    restart_opt: Whether to run hyperparameter optimization from scratch
    model_folder: Folder path where models are serialized
    hyperparam_iterations: Number of iterations of random search
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data fromatter (see
      expt_settings.dataformatter.GenericDataFormatter)
  """

  if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))

  default_keras_session = tf.compat.v1.keras.backend.get_session()

  if use_gpu:
    tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id='0')
  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")

  print("### Running hyperparameter optimization for {} ###".format(expt_name))
  print("Loading & splitting data...")
  raw_data = pd.read_csv('/home/duvvuri.b/ondemand/data/sys/myjobs/projects/default/21/monthly_streamflow.csv', index_col=0)
  sub_comids = pd.read_csv('/home/duvvuri.b/ondemand/data/sys/myjobs/projects/default/21/sel_comids.csv')
  raw_data = raw_data[raw_data['COMID'].isin(sub_comids['COMIDS'])]
  raw_data[['clz_cl_smj', 'fec_cl_smj','lit_cl_smj','fmh_cl_smj','tbi_cl_smj','wet_cl_smj','month']] =  raw_data[['clz_cl_smj', 'fec_cl_smj','lit_cl_smj','fmh_cl_smj','tbi_cl_smj','wet_cl_smj','month']].astype('int')

  train, valid, test = data_formatter.split_data(raw_data)
  train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

  # Sets up default params
  fixed_params = data_formatter.get_experiment_params()
  param_ranges = data_formatter.get_default_model_params()#ModelClass.get_hyperparm_choices()
  fixed_params["model_folder"] = model_folder

  print("*** Loading hyperparm manager ***")
  opt_manager = HyperparamOptManager({k: [param_ranges[k]] for k in param_ranges}, fixed_params, model_folder)
#   opt_manager = HyperparamOptManager(param_ranges, fixed_params, model_folder,worker_number=1, search_iterations=40)

  success = opt_manager.load_results()
  if success and not restart_opt:
    print("Loaded results from previous training")
  else:
    print("Creating new hyperparameter optimisation")
    opt_manager.clear()

  print("*** Running calibration ***")
  while len(opt_manager.results.columns) < 1:
    print("# Running hyperparam optimisation {} of {} for {}".format(
        len(opt_manager.results.columns) + 1, hyperparam_iterations, "TFT"))

    tf.compat.v1.reset_default_graph()
    with tf.Graph().as_default(), tf.compat.v1.Session(config=tf_config) as sess:

      tf.compat.v1.keras.backend.set_session(sess)

      params = opt_manager.get_next_parameters()
      print('params')
      print([j for e,j in enumerate(params.items()) if e<7])
      model = ModelClass(params, use_cudnn=use_gpu)

      if not model.training_data_cached():
        model.cache_batched_data(train, "train", num_samples=train_samples)
        model.cache_batched_data(valid, "valid", num_samples=valid_samples)

      sess.run(tf.compat.v1.global_variables_initializer())
      model.fit()

      val_loss = model.evaluate()

      if np.allclose(val_loss, 0.) or np.isnan(val_loss):
        # Set all invalid losses to infintiy.
        # N.b. val_loss only becomes 0. when the weights are nan.
        print("Skipping bad configuration....")
        val_loss = np.inf

      opt_manager.update_score(params, val_loss, model)

      tf.compat.v1.keras.backend.set_session(default_keras_session)

  print("*** Running tests ***")
  tf.reset_default_graph()
  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.compat.v1.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
        
        model.load(opt_manager.hyperparam_folder)
    
        print("Computing best validation loss")
        val_loss = model.evaluate(valid)

        def extract_numerical_data(data):
            """Strips out forecast time and identifier columns."""
            return data[[ col for col in data.columns if col not in {"forecast_time", "identifier"} ]]
        
        # return model,data_formatter,utils,test,all_data
        print("Computing test loss")
        output_map = model.predict(test, return_targets=True)
    
        targets = data_formatter.format_predictions(output_map["targets"])
        forecasts = data_formatter.format_predictions(output_map["predictions"])
        tf.compat.v1.keras.backend.set_session(default_keras_session)

  print("Training completed @ {}".format(dte.datetime.now()))
  print("Best validation loss = {}".format(val_loss))
  print("Params:")
  return targets,forecasts,opt_manager


if __name__ == "__main__":

    ExperimentConfig = expt_settings.configs.ExperimentConfig
    HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
    ModelClass = libs.tft_model.TemporalFusionTransformer
    tf.experimental.output_all_intermediates(True)
    experiment_names = ExperimentConfig.default_experiments

    # Load settings for default experiments
    # name, folder, use_tensorflow_with_gpu, restart = get_args()
    name, folder, use_tensorflow_with_gpu, restart = 'streamflow',  '/home/duvvuri.b/ondemand/data/sys/myjobs/projects/default/23/' ,True,True
    print("Using output folder {}".format(folder))
    
    config = ExperimentConfig(name, folder)
    formatter = config.make_data_formatter()
    
    # Customise inputs to main() for new datasets.
    targets,forecasts,opt_manager = main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      restart_opt=restart,
      model_folder=os.path.join(config.model_folder, "fixed_hyper2_2"),
      hyperparam_iterations=config.hyperparam_iterations,
      data_csv_path=config.data_csv_path,
      data_formatter=formatter)
