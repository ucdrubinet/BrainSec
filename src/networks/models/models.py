"""Model Script Used By train.py and predict.py"""
from tensorflow import keras

from networks.metrics import SparseIoU, SparseMeanIoU, SparsePixelAccuracy, \
        SparseMeanAccuracy, SparseFreqIoU, SparseConfusionMatrix
from networks.losses import BalancedSCCE, SparseFocalLoss, BalancedSparseFocalLoss

from .FCN import fcn_model
from .UNet import unet_model_no_pad, unet_model_zero_pad

def get_model(model_name: str) -> keras.Model:
    """Get the keras.Model based on input"""
    if model_name == 'UNet_No_Pad':
        return unet_model_no_pad(output_channels=3)
    if model_name == 'UNet_No_Pad_3Layer':
        return unet_model_no_pad(output_channels=3,
                                 unet_levels=3,
                                 model_name='UNet_No_Pad_3Layer')
    if model_name == 'UNet_Zero_Pad':
        return unet_model_zero_pad(output_channels=3)
    if model_name == 'UNet_Zero_Pad_3Layer':
        return unet_model_zero_pad(output_channels=3,
                                   unet_levels=3,
                                   model_name='UNet_Zero_Pad_3Layer')
    if model_name == 'UNet_Zero_Pad_2019O':
        return unet_model_zero_pad(output_channels=3,
                                   unet_levels=4,
                                   first_level_filters=32,
                                   kernel_size=(5, 5),
                                   kernel_initializer='he_uniform',
                                   activation='elu',
                                   use_dropout=True,
                                   dropout_rate=0.2,
                                   model_name='UNet_Zero_Pad_2019O')
    if model_name == 'FCN':
        return fcn_model(classes=3, bn=True)
    raise ValueError('Unknown model')

def load_whole_model(ckpt_filepath: str) -> keras.Model:
    """Load whole model from checkpoint
    This includes:
        * Model's architecture/config
        * Model's weight values
        * Model's compilation information (if compile()) was called
        * Optimizer and its state, if any
    """
    return keras.models.load_model(
        ckpt_filepath,
        custom_objects={
            'BalancedSCCE': BalancedSCCE,
            'SparseFocalLoss': SparseFocalLoss,
            'BalancedSparseFocalLoss': BalancedSparseFocalLoss,
            'SparseIoU': SparseIoU,
            'SparseMeanIoU': SparseMeanIoU,
            'SparsePixelAccuracy': SparsePixelAccuracy,
            'SparseMeanAccuracy': SparseMeanAccuracy,
            'SparseFreqIoU': SparseFreqIoU,
            'SparseConfusionMatrix': SparseConfusionMatrix})
