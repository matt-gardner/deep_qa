from typing import Dict, List, Union

from overrides import overrides
import numpy

from . import Field
from .. import Vocabulary

class LabelField(Field):
    """
    A ``LabelField`` is a categorical label of some kind, where the labels are either strings of
    text or 0-indexed integers.  If the labels need indexing, we will use a :class:`Vocabulary` to
    convert the string labels into integers.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default=``labels``)
        The namespace to use for converting label strings into integers.  If you have multiple
        different label fields in your data, you should make sure you use different namespaces for
        each one.
    index_labels : ``bool``, optional (default=``True``)
        Do the labels need indexing?  If they are categorical labels, then this should be ``True``.
        If they are already 0-indexed integers, set this to ``False``.
    """
    def __init__(self, label: Union[str, int], label_namespace: str='labels', index_labels: bool=True):
        self._label = label
        self._label_namespace = label_namespace
        if index_labels:
            self._label_id = None
        else:
            assert isinstance(label, int), "Labels must be ints if you want to skip indexing"
            self._label_id = label

    @overrides
    def needs_indexing(self):
        return self._label_id is None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        counter[self._label_namespace][self._label] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        self._label_id = vocab.get_token_index(self._label, self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_labels': self._label_id + 1}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        label_array = numpy.zeros(padding_lengths['num_labels'])
        label_array[self._label_id] = 1
        return [label_array]

    @overrides
    def empty_field(self):
        return LabelField(0, 'labels', False)
