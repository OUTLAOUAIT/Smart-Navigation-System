"""This module contains constants used in the command line interface"""
# color spaces
GRAY = 'gray'
HSV = 'hsv'
LAB = 'lab'
RGB = 'rgb'

# image channels
IMAGE_CHANNELS = '0,1,2'

# image matching operations
FDM = 'fdm'
HM = 'hm'

# match proportion range
MATCH_ZERO = 0.0
MATCH_FULL = 1.0

DIM_1 = 1
DIM_2 = 2
DIM_3 = 3

HM_PLOT_FILE = 'hm_plot.png'


"""This module defines the ColorSpaceConverter interface"""
import cv2
import abc
from typing import NamedTuple, Tuple, Any, Dict
import numpy as np
import os
from functools import wraps


ChannelRange = NamedTuple('ChannelRange', [('min', float), ('max', float)])

"""This module defines class for storing command line parameters"""
from keyword import iskeyword


class Params:
    """ Params class stores command line parameters as dynamic attributes """

    def __init__(self, mapping: Dict[str, Any]):
        self.__data = dict()

        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError(f'parameter name must be {repr(str)}')

            # check whether key is a valid Python identifier
            if not str.isidentifier(key):
                raise NameError(f'wrong name for an attribute: {key}')

            # check whether key does not collide with python keywords
            if iskeyword(key):
                raise NameError(f'Python keyword {key} cannot be used')

            self.__data[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self.__data:
            return self.__data[name]
        raise AttributeError(f'there is no attribute {name}')

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}({self.__data})'

    def __str__(self) -> str:
        params_str = str()
        for key, value in self.__data.items():
            params_str += f'{key} : {value}\n'
        return params_str


class ColorSpaceConverter(abc.ABC):
    """ the ColorSpaceConverter interface declares operations common to all
    color space conversion algorithms """

    @abc.abstractmethod
    def convert(self, image: np.ndarray) -> np.ndarray:
        """ converts image from source to target color space """

    @abc.abstractmethod
    def convert_back(self, image: np.ndarray) -> np.ndarray:
        """ converts image from target color space back
        to source color space """

    @abc.abstractmethod
    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        """ returns the ranges of the color space """


"""This module provides a no-op implementation that is used when no color
space conversion is requested"""

RGB_MIN = 0.0
RGB_MAX = 1.0


class IdentityConverter(ColorSpaceConverter):
    """ IdentityConverter just gives back the original image color space """

    def __init__(self, dim: int = DIM_3):
        self._dim = int(dim)

    def convert(self, image: np.ndarray) -> np.ndarray:
        return image

    def convert_back(self, image: np.ndarray) -> np.ndarray:
        return image

    def target_channel_ranges(self) -> Tuple[ChannelRange, ...]:
        return tuple([ChannelRange(RGB_MIN, RGB_MAX)] * self._dim)


"""This module contains a function that creates a specified color space
converter"""


def build_cs_converter(color_space: str) -> ColorSpaceConverter:
    """Creates ColorSpaceConverter for specified color space"""
    target_color_space = color_space.lower()

    return IdentityConverter()


"""This module defines Operation interface"""



ChannelsType = Tuple[int, ...]

_FLOAT_TYPES = (np.float32, np.float64)


class Operation(abc.ABC):
    """ The interface declares operations common to all operations """

    # pylint: disable=too-few-public-methods (R0903)

    def __init__(self, channels: ChannelsType, check_input: bool = False):
        self.channels = channels
        self.check_input = check_input

    @property
    def channels(self) -> ChannelsType:
        """ Returns the channels of the color space """
        return self._channels

    @channels.setter
    def channels(self, channels: ChannelsType) -> None:
        if not isinstance(channels, tuple):
            raise TypeError(f'channels has to be of type {repr(ChannelsType)}')

        for channel in channels:
            if not isinstance(channel, int):
                raise TypeError(
                    f'Each element of channels has to be of type {repr(int)}')
        self._channels = channels

    @property
    def check_input(self) -> bool:
        """ Returns the input check flag """
        return self._check_input

    @check_input.setter
    def check_input(self, check_input: bool) -> None:
        self._check_input = bool(check_input)

    def _verify_input(self, source: np.ndarray,
                      reference: np.ndarray) -> None:
        if source.ndim != reference.ndim:
            raise ValueError(
                f'Source and reference have to be of the same dimension, '
                f'but source has {source.ndim} and reference has '
                f'{reference.ndim}')
        if source.ndim != DIM_3:
            raise ValueError(
                f'Input images have to be 3 dimensional, but '
                f'they are {source.ndim} dimensional')
        if source.shape[-1] != reference.shape[-1]:
            raise ValueError(
                f'The number of channels in source and reference '
                f'have to be the same, but source has '
                f'{source.shape[-1]} and reference has '
                f'{reference.shape[-1]}')
        if source.dtype not in _FLOAT_TYPES:
            raise TypeError(f'Source has to be of float type,'
                            f'but it is {source.dtype}')
        if reference.dtype not in _FLOAT_TYPES:
            raise TypeError(f'Reference has to be of float type, '
                            f'but it is {reference.dtype}')

        for idx, channel in enumerate(self._channels):
            if abs(channel) >= source.shape[-1]:
                raise IndexError(
                    f'{idx} channel is out of range')

    def __call__(self, source: np.ndarray,
                 reference: np.ndarray) -> np.ndarray:
        """ Calls operation implementation """
        if self.check_input:
            self._verify_input(source, reference)
        return self._apply(source, reference)

    @abc.abstractmethod
    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:
        """ Operation implementation """


"""This module defines Context for launching matching operations"""

class OperationContext:
    """ This class executes a given matching operation with a given
    color space converter """

    def __init__(self, converter: ColorSpaceConverter,
                 operation: Operation) -> None:
        self.converter = converter
        self.operation = operation

    @property
    def converter(self) -> ColorSpaceConverter:
        """ Returns the color converter """
        return self._converter

    @converter.setter
    def converter(self, converter: ColorSpaceConverter) -> None:
        if isinstance(converter, ColorSpaceConverter):
            self._converter = converter
        else:
            raise TypeError(
                f'converter has to be of {repr(ColorSpaceConverter)} type')

    @property
    def operation(self) -> Operation:
        """ Returns the matching operation """
        return self._operation

    @operation.setter
    def operation(self, operation: Operation) -> None:
        if isinstance(operation, Operation):
            self._operation = operation
        else:
            raise TypeError(
                f'converter has to be of {repr(Operation)} type')

    def __call__(self, source: np.ndarray,
                 reference: np.ndarray) -> np.ndarray:
        """ Operation process flow """
        source = self.converter.convert(source)
        reference = self.converter.convert(reference)
        result = self.operation(source, reference)
        result = self.converter.convert_back(result)
        return result


"""This module implements Feature Distribution Matching operation"""


class FeatureDistributionMatching(Operation):
    """Feature Distribution Matching operation class"""

    # pylint: disable=too-few-public-methods (R0903)

    def __init__(self, channels: ChannelsType,
                 channel_ranges: Tuple[ChannelRange, ...],
                 check_input: bool = False):
        super().__init__(channels, check_input)
        self.channel_ranges = channel_ranges

    @property
    def channel_ranges(self) -> Tuple[ChannelRange, ...]:
        """ Returns the channels of the color space """
        return self._channel_ranges

    @channel_ranges.setter
    def channel_ranges(self, channel_ranges: Tuple[ChannelRange, ...]) -> None:
        if not isinstance(channel_ranges, tuple):
            raise TypeError(
                f'channel ranges has to be of type '
                f'{repr(Tuple[ChannelRange, ...])}')

        for channel_range in channel_ranges:
            if not isinstance(channel_range, ChannelRange):
                raise TypeError(
                    f'Channel range has to be of {repr(ChannelRange)}')
        self._channel_ranges = channel_ranges

    def _apply(self, source: np.ndarray,
               reference: np.ndarray) -> np.ndarray:

        matching_result = self._matching(source[:, :, self.channels],
                                         reference[:, :, self.channels])

        result = np.copy(source)
        # Replace selected channels with matching result
        result[:, :, self.channels] = matching_result

        # Replace selected channels
        for channel in self.channels:
            result[:, :, channel] = np.clip(result[:, :, channel],
                                            self.channel_ranges[
                                                channel].min,
                                            self.channel_ranges[
                                                channel].max)

        return result.astype(np.float32)

    @staticmethod
    def _matching(source: np.ndarray,
                  reference: np.ndarray) -> np.ndarray:
        """ Run all transformation steps """
        # 1.) reshape to feature matrix (H*W,C)
        feature_mat_src = FeatureDistributionMatching._get_feature_matrix(
            source)
        feature_mat_ref = FeatureDistributionMatching._get_feature_matrix(
            reference)

        # 2.) center (subtract mean)
        feature_mat_src, _ = FeatureDistributionMatching._center_image(
            feature_mat_src)
        feature_mat_ref, reference_mean = \
            FeatureDistributionMatching._center_image(feature_mat_ref)

        # 3.) whitening: cov(feature_mat_src) = I
        feature_mat_src_white = FeatureDistributionMatching._whitening(
            feature_mat_src)

        # 4.) transform covariance: cov(feature_mat_ref) = covariance_ref
        feature_mat_src_transformed = \
            FeatureDistributionMatching._covariance_transformation(
                feature_mat_src_white, feature_mat_ref)

        # 5.) Add reference mean
        feature_mat_src_transformed += reference_mean

        # 6.) Reshape
        result = feature_mat_src_transformed.reshape(source.shape)

        return result

    @staticmethod
    def _get_feature_matrix(image: np.ndarray) -> np.ndarray:
        """ Reshapes an image (H, W, C) to
        a feature vector (H * W, C)
        :param image: H x W x C image
        :return feature_matrix: N x C matrix with N samples and C features
        """
        feature_matrix = np.reshape(image, (-1, image.shape[-1]))
        return feature_matrix

    @staticmethod
    def _center_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Centers the image by removing mean
        :returns centered image and original mean
        """
        image = np.copy(image)
        image_mean = np.mean(image, axis=0)
        image -= image_mean
        return image, image_mean

    @staticmethod
    def _whitening(feature_mat: np.ndarray) -> np.ndarray:
        """
        Transform the feature matrix so that cov(feature_map) = Identity or
        if the feature matrix is one dimensional so that var(feature_map) = 1.
        :param feature_mat: N x C matrix with N samples and C features
        :return feature_mat_white: A corresponding feature vector with an
        identity covariance matrix or variance of 1.
        """
        if feature_mat.shape[1] == 1:
            variance = np.var(feature_mat)
            feature_mat_white = feature_mat / np.sqrt(variance)
        else:
            data_cov = np.cov(feature_mat, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(data_cov)
            sqrt_s = np.diag(np.sqrt(s_vec))
            feature_mat_white = (feature_mat @ u_mat) @ np.linalg.inv(sqrt_s)
        return feature_mat_white

    @staticmethod
    def _covariance_transformation(feature_mat_white: np.ndarray,
                                   feature_mat_ref: np.ndarray) -> np.ndarray:
        """
        Transform the white (cov=Identity) feature matrix so that
        cov(feature_mat_transformed) = cov(feature_mat_ref). In the 2d case
        this becomes:
        var(feature_mat_transformed) = var(feature_mat_ref)
        :param feature_mat_white: input with identity covariance matrix
        :param feature_mat_ref: reference feature matrix
        :return: feature_mat_transformed with cov == cov(feature_mat_ref)
        """
        if feature_mat_white.shape[1] == 1:
            variance_ref = np.var(feature_mat_ref)
            feature_mat_transformed = feature_mat_white * np.sqrt(variance_ref)
        else:
            covariance_ref = np.cov(feature_mat_ref, rowvar=False)
            u_mat, s_vec, _ = np.linalg.svd(covariance_ref)
            sqrt_s = np.diag(np.sqrt(s_vec))

            feature_mat_transformed = (feature_mat_white @ sqrt_s) @ u_mat.T
        return feature_mat_transformed


"""This module contains a function that creates the operation context"""
#from core import FDM, HM, Params
#from utils.cs_conversion.cs_converter_builder import build_cs_converter

#from . import Operation
#from .operation_context import OperationContext
#from .operations import FeatureDistributionMatching, HistogramMatching


def build_operation_context(matching_type: str,
                            params: Params) -> OperationContext:
    """Creates OperationContext for specified matching type and params"""
    converter = build_cs_converter(params.color_space)
    channel_ranges = converter.target_channel_ranges()

    channels = tuple(int(c) for c in params.channels.split(','))

    operation = \
        FeatureDistributionMatching(channels,
                                    check_input=params.verify_input,
                                    channel_ranges=channel_ranges)

    return OperationContext(converter, operation)




#from core import (FDM, GRAY, HM, HSV, IMAGE_CHANNELS, LAB, MATCH_FULL,
                  #MATCH_ZERO,RGB, Params)
#from utils import application


#from core import DIM_1, GRAY, HM, HM_PLOT_FILE, Params
#from matching.operation_context_builder import build_operation_context


MAX_VALUE_8_BIT = 255


'''def read_image(path: str) -> np.ndarray:
    """ This function reads an image and transforms it to RGB color space """
    if os.path.exists(path):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32) / MAX_VALUE_8_BIT
        if image.ndim == DIM_2:
            return image[:, :, np.newaxis]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError(f'Invalid image path {path}')'''

def read_image(image: np.ndarray) -> np.ndarray:
    """ This function reads an image and transforms it to RGB color space """
    #image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    image = image.astype(np.float32) / MAX_VALUE_8_BIT
    if image.ndim == DIM_2:
        return image[:, :, np.newaxis]
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB);

    #raise ValueError(f'Invalid image path {path}')

def write_image(image: np.ndarray, path: str) -> None:
    """ This function transforms an image to BGR color space
    and writes it to disk """
    if image.dtype == np.float32:
        image = (image * MAX_VALUE_8_BIT).astype(np.uint8)
    if image.dtype == np.uint8:
        if image.shape[-1] == DIM_1:
            output_image = image[:, :, 0]
        else:
            output_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return output_image
        '''if not cv2.imwrite(path, output_image):
            raise ValueError(
                f'Output directory {os.path.dirname(path)} does not exist')'''
    '''else:
        raise TypeError(
            f'Cannot write image with type {image.dtype}')'''

"""This module provides a function to perform the matching operation"""




def run(operation_type: str, params: Params) -> None:
    """
    This function gets the input images, creates the specified
    OperationContext, uses it and saves the resulting
    image
    """

    source = read_image(params.source_path)
    reference = read_image(params.reference_path)
    
    #color_check(params.color_space, source, reference)

    op_ctx = build_operation_context(operation_type, params)
    result = op_ctx(source, reference)

    #return result
    
    return write_image(result, params.result_path)




def feature_distribution_matching(source_path: str, reference_path: str,
                                   result_path: str, color_space: str = RGB,
                                   channels: str = IMAGE_CHANNELS,
                                   plot: bool = False,
                                   verify_input: bool = False) -> None:
    """
    Applies feature distribution matching to an image.
    :param source_path: path to the source image
    :param reference_path: path to the reference image
    :param result_path: path to save the result image
    :param color_space: color space for the image (RGB, HSV, LAB, GRAY)
    :param channels: channels to match (e.g. '0,1', '1', '0,2', '0,1,2')
    :param plot: whether to plot the result
    :param verify_input: whether to verify input data
    """

    # Set up parameters
    params = Params({
        'source_path': source_path,
        'reference_path': reference_path,
        'result_path': result_path,
        'color_space': color_space,
        'channels': channels,
        'plot': plot,
        'verify_input': verify_input,
    })

    #print(params)

    # Apply feature distribution matching
    fdm = run(FDM, params)
    return fdm
    #fdm._apply(params)

    #Plot result
  
'''feature_distribution_matching('data/munich_1.png',
                               'data/munich_2.png',
                               'output3.png')
'''

'''image_source = cv2.imread("data/munich_1.png")
image_reference = cv2.imread("data/munich_2.png")


feature_distribution_matching(image_source,
                              image_reference,
                               'output5.png')'''
