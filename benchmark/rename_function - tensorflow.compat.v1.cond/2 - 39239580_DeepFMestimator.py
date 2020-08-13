import tensorflow as tf
from tensorflow import estimator
# from tensorflow.contrib.layers.python.layers import l2_regularizer
from tensorflow.keras.regularizers import l2
# from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.compat.v1.layers import batch_normalization



# 参考相应的文档
class DeepFMEstimator(object):
    def __init__(self,config, model_params,model_dir):
        self.config = config
        self.model_params = model_params
        self.model = estimator.Estimator(model_fn=self.my_model,
                                         model_dir=model_dir,
                                         params=self.model_params,
                                         config=self.config
                                         )

    def my_model(self,features, labels, mode, params):
        # 超参数
        field_size = params["field_size"]
        feature_size = params["feature_size"]
        embedding_size = params["embedding_size"]
        l2_reg = params["l2_reg"]
        learning_rate = params["learning_rate"]
        module_type = params["module_type"]
        batch_norm = params["batch_norm"]
        deep_layers = list(map(int, params["deep_layers"].split(",")))
        dropout_fm = list(map(float, params["dropout_fm"].split(",")))
        dropout_dnn = list(map(float, params["dropout_dnn"].split(",")))
        batch_norm_decay = params["batch_norm_decay"]

        if module_type == "DeepFM":
            concat_size = field_size+embedding_size+deep_layers[-1]
        elif module_type == "FM":
            concat_size = field_size+embedding_size
        elif module_type == "Deep":
            concat_size = field_size+deep_layers[-1]
        else:
            raise ValueError("please Enter correct module type!")
        # 创建权重
        fm_b = tf.compat.v1.get_variable(name="fm_b", shape=[feature_size, 1], initializer=tf.compat.v1.glorot_normal_initializer())
        fm_w = tf.compat.v1.get_variable(name="fm_w",shape=[feature_size, embedding_size], initializer=tf.compat.v1.glorot_normal_initializer())
        concat_w = tf.compat.v1.get_variable(name="concat_w", shape=[concat_size, 1],initializer=tf.compat.v1.glorot_normal_initializer())
        concat_b = tf.compat.v1.get_variable(name="concat_b", shape=[1], initializer=tf.compat.v1.constant_initializer(0.01))
        # 构建特征
        feat_id = features["feat_ids"]
        feat_ids = tf.compat.v1.reshape(feat_id, shape=[-1, field_size])   # [None, F]
        feat_values = features["feat_value"]
        feat_values = tf.compat.v1.reshape(feat_values, shape=[-1, field_size, 1])  # [None, F, 1]
        embeddings = tf.compat.v1.nn.embedding_lookup(fm_w, feat_id)   # [None, k]
        embeddings = tf.compat.v1.multiply(embeddings, feat_values)   #

        # 构建模型
        # -----------------------------first-order-term------------------------------------
        with tf.compat.v1.variable_scope("First_order_term"):
            y_first_order = tf.compat.v1.nn.embedding_lookup(fm_b, feat_ids)
            y_first_order = tf.compat.v1.reduce_sum(tf.compat.v1.multiply(y_first_order, feat_values), axis=2)
            y_first_order = tf.compat.v1.nn.dropout(y_first_order, rate=1-dropout_fm[0])

        # -----------------------------second-order-term-----------------------------------
        with tf.compat.v1.variable_scope("Second_order_term"):
            # ---------------------sum_quare part--------------------------
            summed_features_emb = tf.compat.v1.reduce_sum(embeddings, axis=1)
            summed_features_emb_square = tf.compat.v1.square(summed_features_emb)
            # ---------------------suqre_sum part--------------------------
            squared_feature_emb = tf.compat.v1.square(embeddings)
            squared_sum_feature_emb = tf.compat.v1.reduce_sum(squared_feature_emb, axis=1)

            # second order
            y_second_order = 0.5* tf.compat.v1.subtract(summed_features_emb_square,
                                                        squared_sum_feature_emb)

        # ------------------------------ deep part -------------------------------
        with tf.compat.v1.variable_scope("Deep-part"):
            y_deep = tf.compat.v1.reshape(embeddings, shape=[-1, field_size* embedding_size])
            if mode == tf.compat.v1.estimator.ModeKeys.TRAIN:
                y_deep = tf.compat.v1.nn.dropout(y_deep, rate=1-dropout_dnn[0])
            for i in range(len(deep_layers)):
                y_deep = tf.compat.v1.layers.dense(y_deep,deep_layers[i], activation=tf.compat.v1.nn.relu,
                                                   kernel_regularizer=l2(l2_reg))
                if batch_norm:
                    y_deep =self.batch_norm_layer(y_deep, batch_norm_decay=batch_norm_decay, train_phase=True, scope_bn="bn_%d"%i)
                if mode == tf.estimator.ModeKeys.TRAIN:
                    y_deep = tf.compat.v1.nn.dropout(y_deep, rate=dropout_dnn[i+1])

        with tf.compat.v1.variable_scope("Out"):
            if module_type == "DeepFM":
                concat_input = tf.compat.v1.concat([y_first_order, y_second_order, y_deep], axis=1)
            elif module_type == "FM":
                concat_input = tf.compat.v1.concat([y_first_order, y_second_order], axis=1)
            elif module_type == "Deep":
                concat_input = y_deep
            else:
                raise ValueError("Error use module_type")
            logits= tf.compat.v1.add(tf.compat.v1.matmul(concat_input, concat_w), concat_b)   # 输出的节点个数为１

        # logits = tf.compat.v1.layers.dense(concat_input, untis=1)  # 最后为一个节点的数据
        pred = tf.compat.v1.sigmoid(logits)
        predictions = {"prob": pred}
        export_outputs = {
            tf.compat.v1.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                estimator.export.PredictOutput(predictions)
        }

        # 预测的值
        if mode == estimator.ModeKeys.PREDICT:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )

        loss = tf.compat.v1.reduce_mean(
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                              labels = tf.compat.v1.cast(
                                                                  deep_layers, dtype=tf.compat.v1.float32)))

        eval_metric_ops = {
            "AUC": tf.compat.v1.metrics.auc(labels=labels, predictions=pred),
            "Accuracy": tf.compat.v1.metrics.accuracy(labels=labels, predictions=pred)
        }

        if mode == estimator.ModeKeys.EVAL:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                eval_metric_ops=eval_metric_ops
            )

        # 训练中要用到的优化器
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=params["learning_rate"])
        train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.global_step())
        if mode == estimator.ModeKeys.TRAIN:
            return estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op
            )

    def train(self, filename, batch_size, input_fn, num_epoches, need_shuffle=True,
              hooks=None, steps=None, max_steps=None, saving_listeners=None):
        self.model.train(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                         hooks=hooks,
                         steps=steps,
                         max_steps=max_steps,
                         saving_listeners=saving_listeners)

    def predict(self, filename, batch_size, input_fn, num_epoches=1, need_shuffle=True,
                predict_keys=None, hooks=None, checkpoint_path=None,
                yield_single_examples=True):
        predict = self.model.predict(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                                     predict_keys=predict_keys,
                                     hooks=hooks,
                                     checkpoint_path=checkpoint_path,
                                     yield_single_examples=yield_single_examples
                                     )
        return predict

    def evaluate(self, filename, batch_size, input_fn, num_epoches=1, need_shuffle=False,
                 steps=None, hooks=None, checkponit_path=None):
        self.model.evaluate(input_fn=lambda: input_fn(
            filename=filename,
            batch_size=batch_size,
            num_epoches=num_epoches,
            need_shuffle=need_shuffle),
                            steps=steps,
                            hooks=hooks,
                            checkpoint_path=checkponit_path)

    @staticmethod
    def batch_norm_layer(x, batch_norm_decay, train_phase, scope_bn):
        bn_train = batch_normalization(inputs=x, momentum=batch_norm_decay, center=True, scale=True, training=True,
                                       reuse=None, trainable=True, name=scope_bn)
        # bn_train = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
        #                       is_training=True, reuse=None, trainable=True, scope=scope_bn)

        # bn_inference = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
        #                           is_training=False, reuse=True, trainable=True, scope=scope_bn)

        bn_inference = batch_normalization(inputs=x, momentum=batch_norm_decay, center=True, scale=True, training=False,
                                           reuse=True, trainable=True, name=scope_bn)

        z = tf.compat.v1.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
