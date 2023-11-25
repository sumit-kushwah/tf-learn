import pandas as pd
import tensorflow as tf
import os

train_df = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
eval_df = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')


categorical_cols = ['sex', 'n_siblings_spouses',
                    'parch', 'class', 'deck', 'embark_town', 'alone']
numeric_cols = ['age', 'fare']

feature_columns = []
for feature_name in categorical_cols:
    vocabulary = train_df[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

for feature_name in numeric_cols:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))


def make_input_fn(train_df, y_train, epoches=10, batch_size=32, shuffle=True):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(train_df), y_train))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(epoches)
        return ds
    return input_function


train_input_fn = make_input_fn(train_df, y_train)
eval_input_fn = make_input_fn(eval_df, y_eval, epoches=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn)

# Save the model
serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
    tf.feature_column.make_parse_example_spec(feature_columns))
export_path = linear_est.export_saved_model(
    os.getenv('AIP_MODEL_DIR', 'gcs/leap-vertex-ai-im/titanic-tensorflow-linear-model/model_output'), serving_input_fn)

print(linear_est.evaluate(eval_input_fn))

print(tf.__version__)
