from typing import Dict, List

from overrides import overrides
import numpy

from . import IndexedField, UnindexedField
from .. import Vocabulary

class IntegerLabelField(IndexedField):
    """
    An ``IntegerLabelField`` represents a categorical label of some kind, where the labels are
    already converted into integers.  These integers should be 0-indexed, with no skipped values -
    we will create a one-hot representation of these labels when we actually construct arrays.
    """
    def __init__(self, label_id: int):
        self._label_id = label_id

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_labels': self._label_id + 1}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        label_array = numpy.zeros(padding_lengths['num_labels'])
        label_array[self._label_id] = 1
        return [label_array]


class StringLabelField(UnindexedField):
    """
    A ``StringLabelField`` is a categorical label of some kind, where the labels are strings of
    text.  We will use a :class:`Vocabulary` to convert the string labels into integers, giving an
    :class:`IntegerLabelField` in return.

    Parameters
    ----------
    label : ``str``
    label_namespace : ``str``, optional (default=``labels``)
        The namespace to use for converting label strings into integers.  If you have multiple
        different label fields in your data, you should make sure you use different namespaces for
        each one.
    """
    def __init__(self, label: str, label_namespace: str='labels'):
        self._label = label
        self._label_namespace = label_namespace

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        counter[self._label_namespace][self._label] += 1

    @overrides
    def index(self, vocab: Vocabulary) -> IntegerLabelField:
        return IntegerLabelField(vocab.get_token_index(self._label, self._label_namespace))
