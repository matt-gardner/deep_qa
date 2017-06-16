from typing import Dict, List

from overrides import overrides
import numpy

from . import Field, SequenceField


class IndexField(Field):
    """
    An ``IndexField`` is an index into a :class:`SequenceField`, as might be used for
    representing a correct answer option in a list, or a span begin and span end position in a
    passage, for example.  Because it's an index into a :class:`SequenceField`, we take one of
    those as input and use it to compute padding lengths, so we create a one-hot representation of
    the correct length.
    """
    def __init__(self, index: int, sequence_field: SequenceField):
        self._index = index
        self._sequence_field = sequence_field

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_options': self._sequence_field.sequence_length()}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        one_hot_index = numpy.zeros(padding_lengths['num_options'])
        one_hot_index[self._index] = 1
        return one_hot_index

    @overrides
    def empty_field(self):
        return IndexField(0, None)
