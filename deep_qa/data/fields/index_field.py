from typing import Dict, List

from overrides import overrides
import numpy

from . import Field
from .text_field import TextField


class IndexField(Field):
    """
    An ``IndexField`` is an index into some piece of tokenized text, as might be used for
    representing a span begin and span end position in a passage, for example.  Because it's an
    index into a :class:`TextField`, we take one of those as input and use it to compute padding
    lengths, so we create a one-hot representation of the correct length.
    """
    def __init__(self, index: int, text_field: TextField):
        self._index = index
        self._text_field = text_field

    @overrides
    def needs_indexing(self):
        return False

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {'num_tokens': self._text_field.get_padding_lengths()['num_tokens']}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        one_hot_index = numpy.zeros(padding_lengths['num_tokens'])
        one_hot_index[self._index] = 1
        return one_hot_index
