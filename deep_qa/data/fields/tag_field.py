from typing import Dict, List

from overrides import overrides
import numpy

from . import Field, SequenceField
from .. import Vocabulary
from ...common.util import pad_sequence_to_length


class TagField(Field):
    """
    A ``TagField`` assigns a categorical label to each element in a :class:`SequenceField`.
    Because it's a labeling of some other field, we take that field as input here, and we use it to
    determine our padding and other things.
    """
    def __init__(self, tags: List[str], sequence_field: SequenceField, tag_namespace: str='tags'):
        self._tags = tags
        self._sequence_field = sequence_field
        self._tag_namespace = tag_namespace
        self._indexed_tags = None

    @overrides
    def needs_indexing(self):
        return self._indexed_tags is None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for tag in self._tags:
            counter[self._tag_namespace][tag] += 1

    @overrides
    def index(self, vocab: Vocabulary):
        self._indexed_tags = [vocab.get_token_index(tag, self._tag_namespace) for tag in self._tags]

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        num_tokens = self._sequence_field.sequence_length(self._sequence_field.get_padding_lengths())
        return {'num_tokens': num_tokens, 'num_tags': max(self._indexed_tags)}

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        desired_num_tokens = padding_lengths['num_tokens']
        padded_tags = pad_sequence_to_length(self._indexed_tags, desired_num_tokens, default_value=0)
        one_hot_tags = []
        for tag in padded_tags:
            one_hot_tag = [0] * padding_lengths['num_tags']
            one_hot_tag[tag] = 1
            one_hot_tags.append(one_hot_tag)
        return numpy.asarray(one_hot_tags)

    @overrides
    def empty_field(self):
        # pylint: disable=protected-access
        tag_field = TagField([], None)
        tag_field._indexed_tags = []
        return tag_field
