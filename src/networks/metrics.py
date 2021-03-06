"""Custom metrics based on tf.keras.metrics"""
# pylint: disable=too-many-ancestors

from typing import List

import tensorflow as tf
from tensorflow.keras import metrics

class SparseMeanIoU(metrics.MeanIoU):
    """Calculate Mean IoU"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

class SparseConfusionMatrix(metrics.MeanIoU):
    """Computes confusion matrix"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.total_cm

class SparseIoU(metrics.MeanIoU):
    """Calculate per-class IoU"""
    def __init__(self, class_idx, num_classes, name=None, dtype=None):
        super(SparseIoU, self).__init__(num_classes=num_classes,
                                        name=name, dtype=dtype)

        self.class_idx = class_idx    # Computes this IoU only
        assert self.class_idx >= 0 and self.class_idx < num_classes, \
                f"Invalid class_idx {self.class_idx} " \
                f"for num_classes {self.num_classes}"

    @staticmethod
    def get_iou_metrics(num_classes, class_names) -> List["SparseIoU"]:
        """Static method for getting all per-class IoUs"""
        return [SparseIoU(class_idx=i, num_classes=num_classes,
                          name=f'IoU/Class{i}_{class_names[i]}')
                for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    def result(self):
        sum_over_row = tf.cast(
            tf.math.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.math.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return iou[self.class_idx]

    def get_config(self):
        config = {'class_idx': self.class_idx}
        base_config = super(SparseIoU, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    # pylint: enable=unexpected-keyword-arg, no-value-for-parameter

class SparsePixelAccuracy(metrics.MeanIoU):
    """Calculate Pixel Accuracy"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    def result(self):
        sum_all = tf.cast(
            tf.math.reduce_sum(self.total_cm), dtype=self._dtype)
        true_positive_sum = tf.cast(tf.math.reduce_sum(
            tf.linalg.diag_part(self.total_cm)), dtype=self._dtype)

        return tf.math.divide_no_nan(true_positive_sum, sum_all)
    # pylint: enable=unexpected-keyword-arg, no-value-for-parameter

class SparseMeanAccuracy(metrics.MeanIoU):
    """Calculate Mean Accuracy"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    def result(self):
        sum_over_col = tf.cast(
            tf.math.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.math.reduce_sum(
            tf.cast(tf.math.not_equal(sum_over_col, 0), dtype=self._dtype))

        accuracy = tf.math.divide_no_nan(true_positives, sum_over_col)

        return tf.math.divide_no_nan(
            tf.math.reduce_sum(accuracy, name='mean_acc'), num_valid_entries)
    # pylint: enable=unexpected-keyword-arg, no-value-for-parameter

class SparseFreqIoU(metrics.MeanIoU):
    """Calculate Frequency weighted IoU"""
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

    # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
    def result(self):
        sum_over_row = tf.cast(
            tf.math.reduce_sum(self.total_cm, axis=0), dtype=self._dtype)
        sum_over_col = tf.cast(
            tf.math.reduce_sum(self.total_cm, axis=1), dtype=self._dtype)
        sum_all = tf.cast(
            tf.math.reduce_sum(self.total_cm), dtype=self._dtype)
        true_positives = tf.cast(
            tf.linalg.diag_part(self.total_cm), dtype=self._dtype)

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        iou = tf.math.divide_no_nan(true_positives, denominator)

        weighted_iou = tf.math.multiply_no_nan(iou, sum_over_col)

        return tf.math.divide_no_nan(
            tf.math.reduce_sum(weighted_iou, name='freq_iou'), sum_all)
    # pylint: enable=unexpected-keyword-arg, no-value-for-parameter
# pylint: enable=too-many-ancestors
