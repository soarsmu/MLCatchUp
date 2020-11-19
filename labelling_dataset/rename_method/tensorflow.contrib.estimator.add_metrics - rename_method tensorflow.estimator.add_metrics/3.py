# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
'''Points of Interest(POI) Recommendation
Wide&Deep: https://arxiv.org/pdf/1606.07792.pdf, https://github.com/tensorflow/models/'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
from absl import app as absl_app
from absl import flags
import tensorflow as tf

from official.utils.flags import core as flags_core
from official.utils.logs import hooks_helper
from official.utils.logs import logger
from official.utils.misc import model_helpers

'''
train: 1793396
test: 20000
'''

_CSV_COLUMNS = [
'user_id','user_review_count','friends','average_stars','business_id','neighborhood','city','state',
'postal_code','latitude','longitude','business_review_count','categories','stars','Visited'
]

_CSV_COLUMN_DEFAULTS = [[''], [0], [''], [0.0], [''], [''], [''], [''], 
                        [''], [0.0], [0.0], [0], [''], [0.0], ['']]

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


def define_wide_deep_flags():
    flags_core.define_base()
    flags_core.define_benchmark()
    flags.adopt_module_key_flags(flags_core)

    flags.DEFINE_enum(
        name="model_type", short_name="mt", default="wide_deep",
        enum_values=['wide', 'deep', 'wide_deep'],
        help="Select model type.")

    flags_core.set_defaults(data_dir='data',
                            model_dir='model',
                            train_epochs=60,
                            epochs_between_evals=2,
                            batch_size=300)


def build_model_columns():
    user_review_count = tf.feature_column.numeric_column('user_review_count')
    average_stars = tf.feature_column.numeric_column('average_stars')
    latitude = tf.feature_column.numeric_column('latitude')
    longitude = tf.feature_column.numeric_column('longitude')
    business_review_count = tf.feature_column.numeric_column('business_review_count')
    stars = tf.feature_column.numeric_column('stars')

    user_id = tf.feature_column.categorical_column_with_hash_bucket(
        'user_id', hash_bucket_size=78770) #num users

    business_id = tf.feature_column.categorical_column_with_hash_bucket(
        'business_id', hash_bucket_size=13465) #num business

    neighborhood = tf.feature_column.categorical_column_with_hash_bucket(
        'neighborhood', hash_bucket_size=8000)

    city = tf.feature_column.categorical_column_with_hash_bucket(
        'city', hash_bucket_size=8000)

    state = tf.feature_column.categorical_column_with_hash_bucket(
        'state', hash_bucket_size=8000)
    
    postal_code = tf.feature_column.categorical_column_with_hash_bucket(
        'postal_code', hash_bucket_size=13500)
    
    # multivalent sparse columns
    friends = tf.contrib.layers.sparse_column_with_hash_bucket(
        'friends',
        hash_bucket_size=10000000,
        combiner='sqrtn',
        dtype=tf.string,
        hash_keys=None
    )

    categories = tf.contrib.layers.sparse_column_with_hash_bucket(
        'categories',
        hash_bucket_size=10000000,
        combiner='sqrtn',
        dtype=tf.string,
        hash_keys=None
    )

    base_columns = [
        user_id, business_id, neighborhood, city, state, postal_code, friends, categories
    ]
    
    crossed_columns = [
        tf.feature_column.crossed_column(
            ['user_id', 'neighborhood', 'city', 'state', 'postal_code'], hash_bucket_size=10000000)
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        user_review_count,
        average_stars,
        latitude,
        longitude,
        business_review_count,
        stars,
        tf.feature_column.embedding_column(user_id, dimension=40),
        tf.feature_column.embedding_column(business_id, dimension=30),
        tf.feature_column.embedding_column(neighborhood, dimension=30),
        tf.feature_column.embedding_column(city, dimension=30),
        tf.feature_column.embedding_column(state, dimension=30),
        tf.feature_column.embedding_column(postal_code, dimension=30),
        tf.contrib.layers.embedding_column(friends, dimension=50),
        tf.contrib.layers.embedding_column(categories, dimension=50),
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [100, 75, 50, 25]

  run_config = tf.estimator.RunConfig(
      keep_checkpoint_max = 3,   
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  def precision_at_5(labels, predictions): #custom metric function can only take args such as labels, predictions, etc. define k inside function.
      k = 5
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      precision_at_k = tf.truediv(tp, k)
      return {'precision_at_5': tf.metrics.mean(precision_at_k)}

  def recall_at_5(labels, predictions):
      k = 5
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      tp_plus_fn = tf.reduce_sum(labels)
      recall_at_k = tf.truediv(tp, tp_plus_fn)
      return {'recall_at_5': tf.metrics.mean(recall_at_k)}

  def precision_at_10(labels, predictions):
      k = 10
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      precision_at_k = tf.truediv(tp, k)
      return {'precision_at_10': tf.metrics.mean(precision_at_k)}

  def recall_at_10(labels, predictions):
      k = 10
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      tp_plus_fn = tf.reduce_sum(labels)
      recall_at_k = tf.truediv(tp, tp_plus_fn)
      return {'recall_at_10': tf.metrics.mean(recall_at_k)}

  def precision_at_20(labels, predictions):
      k = 20
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      precision_at_k = tf.truediv(tp, k)
      return {'precision_at_20': tf.metrics.mean(precision_at_k)}

  def recall_at_20(labels, predictions):
      k = 20
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      tp_plus_fn = tf.reduce_sum(labels)
      recall_at_k = tf.truediv(tp, tp_plus_fn)
      return {'recall_at_20': tf.metrics.mean(recall_at_k)}

  def precision_at_30(labels, predictions):
      k = 30
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      precision_at_k = tf.truediv(tp, k)
      return {'precision_at_30': tf.metrics.mean(precision_at_k)}

  def recall_at_30(labels, predictions):
      k = 30
      labels = tf.to_int32(labels)
      prob_for_label_1 = predictions['probabilities'][:,1]
      top_k_values, top_k_indices = tf.nn.top_k(prob_for_label_1, k=k)
      list_tp = tf.gather(labels, top_k_indices)
      tp = tf.reduce_sum(list_tp)
      tp_plus_fn = tf.reduce_sum(labels)
      recall_at_k = tf.truediv(tp, tp_plus_fn)
      return {'recall_at_30': tf.metrics.mean(recall_at_k)}

  if model_type == 'wide':
    estimator = tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
    
  elif model_type == 'deep':
    estimator = tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)

  else:
    estimator = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config,
        dnn_optimizer=tf.contrib.opt.LazyAdamOptimizer(learning_rate=0.01))

  estimator = tf.contrib.estimator.add_metrics(estimator, precision_at_5)
  estimator = tf.contrib.estimator.add_metrics(estimator, recall_at_5)
  estimator = tf.contrib.estimator.add_metrics(estimator, precision_at_10)
  estimator = tf.contrib.estimator.add_metrics(estimator, recall_at_10)
  estimator = tf.contrib.estimator.add_metrics(estimator, precision_at_20)
  estimator = tf.contrib.estimator.add_metrics(estimator, recall_at_20)
  estimator = tf.contrib.estimator.add_metrics(estimator, precision_at_30)
  estimator = tf.contrib.estimator.add_metrics(estimator, recall_at_30)
  return estimator


def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), (
        '%s file is not found.' % data_file)

    def column_string_to_list(list_column):
        sparse_strings = tf.string_split([list_column][1:-1], delimiter=",")
        return tf.SparseTensor(indices=sparse_strings.indices,
                         values=sparse_strings.values,
                         dense_shape=sparse_strings.dense_shape)

    def parse_csv(value):
        print('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        columns[2] = column_string_to_list(columns[2]) #friends
        columns[12] = column_string_to_list(columns[12]) #categories
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('Visited')
        return features, tf.equal(labels, '1')

    dataset = tf.data.TextLineDataset(data_file)

    if shuffle:
        dataset = dataset.shuffle(300000)

    dataset = dataset.map(parse_csv, num_parallel_calls=10)
    # To avoid mixing between separate epochs, repeat after shuffle
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def export_model(model, model_type, export_dir):
    wide_columns, deep_columns = build_model_columns()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn)


def run_wide_deep(flags_obj):
    shutil.rmtree(flags_obj.model_dir, ignore_errors=True)
    model = build_estimator(flags_obj.model_dir, flags_obj.model_type)

    train_file = os.path.join(flags_obj.data_dir, 'train.data')
    test_file = os.path.join(flags_obj.data_dir, 'test.data')

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    def train_input_fn():
        return input_fn(
            train_file, flags_obj.epochs_between_evals, True, flags_obj.batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, flags_obj.batch_size)

    run_params = {
        'batch_size': flags_obj.batch_size,
        'train_epochs': flags_obj.train_epochs,
        'model_type': flags_obj.model_type,
    }

    benchmark_logger = logger.get_benchmark_logger()
    benchmark_logger.log_run_info('wide_deep', 'Yelp POI', run_params,
                                    test_id=flags_obj.benchmark_test_id)

    loss_prefix = LOSS_PREFIX.get(flags_obj.model_type, '')
    train_hooks = hooks_helper.get_train_hooks(
        flags_obj.hooks, batch_size=flags_obj.batch_size,
        tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                        'loss': loss_prefix + 'head/weighted_loss/Sum'})

    # Train and evaluate the model every `flags.epochs_between_evals` epochs.
    for n in range(flags_obj.train_epochs // flags_obj.epochs_between_evals):
        model.train(input_fn=train_input_fn, hooks=train_hooks)
        results = model.evaluate(input_fn=eval_input_fn)

        # Display evaluation metrics
        tf.logging.info('Results at epoch %d / %d',
                        (n + 1) * flags_obj.epochs_between_evals,
                        flags_obj.train_epochs)
        tf.logging.info('-' * 50)

        for key in sorted(results):
            tf.logging.info('%s: %s' % (key, results[key]))

        benchmark_logger.log_evaluation_result(results)

        if model_helpers.past_stop_threshold(
            flags_obj.stop_threshold, results['accuracy']):
            break

    if flags_obj.export_dir is not None:
        export_model(model, flags_obj.model_type, flags_obj.export_dir)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    run_wide_deep(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_wide_deep_flags()
  absl_app.run(main)
