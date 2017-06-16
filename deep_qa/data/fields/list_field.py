from typing import Dict, List

from overrides import overrides
import numpy

from . import Field, SequenceField
from .. import Vocabulary
from ...common.util import pad_sequence_to_length


class ListField(SequenceField):
    """
    A ``ListField`` is a list of other fields.  You would use this to represent, e.g., a list of
    answer options that are themselves ``TextFields``.
    """
    def __init__(self, field_list: List[Field]):
        field_class_set = set([field.__class__ for field in field_list])
        assert len(field_class_set) == 1, "ListFields must contain a single field type, found " +\
                str(field_class_set)
        self._field_list = field_list

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for field in self._field_list:
            field.count_vocab_items(counter)

    @overrides
    def index(self, vocab: Vocabulary):
        for field in self._field_list:
            field.index(vocab)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        field_lengths = [field.get_padding_lengths() for field in self._field_list]
        padding_lengths = {'num_fields': len(self._field_list)}
        for key in field_lengths[0].keys():
            padding_lengths[key] = max(x[key] if key in x else 0 for x in field_lengths)
        return padding_lengths

    @overrides
    def sequence_length(self) -> int:
        return len(self._field_list)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        padded_field_list = pad_sequence_to_length(self._field_list,
                                                   padding_lengths['num_fields'],
                                                   self._field_list[0].empty_field)
        padded_fields = [field.pad(padding_lengths) for field in padded_field_list]
        if isinstance(padded_fields[0], [list, tuple]):
            return [numpy.asarray(x) for x in zip(*padded_fields)]
        else:
            return [numpy.asarray(padded_fields)]

    @overrides
    def empty_field(self):
        raise RuntimeError("Nested ListFields are not implemented, and if you want this "
                           "you should probably try to simplify your data type, anyway")
