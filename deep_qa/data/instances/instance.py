"""
This module contains the base ``Instance`` classes that concrete classes
inherit from. Specifically, there are three classes:

1. ``Instance``, that just exists as a base type with no functionality
2. ``TextInstance``, which adds a ``words()`` method and a method to convert
   strings to indices using a Vocabulary.
3. ``IndexedInstance``, which is a ``TextInstance`` that has had all of its
   strings converted into indices.

This class has methods to deal with padding (so that sequences all have the
same length) and converting an ``Instance`` into a set of Numpy arrays
suitable for use with Keras.

As this codebase is dealing mostly with textual question answering, pretty much
all of the concrete ``Instance`` types will have both a ``TextInstance`` and a
corresponding ``IndexedInstance``, which you can see in the individual files
for each ``Instance`` type.
"""
from typing import Dict, List

import numpy

from ..fields import Field, UnindexedField
from ..vocabulary import Vocabulary


class Instance:
    """
    An ``Instance`` is a collection of :class:`Field` objects, specifying the inputs and outputs to
    some model.  We don't make a distinction between inputs and outputs here, though - all
    operations are done on all fields, and when we return arrays, we return them as dictionaries
    keyed by field name.  A model can then decide which fields it wants to use as inputs as which
    as outputs.

    The ``Fields`` in an ``Instance`` can start out either indexed or un-indexed.  During the data
    processing pipeline, all fields will end up as ``IndexedFields``, and will then be converted
    into padded arrays by a ``DataGenerator``.
    """
    def __init__(self, fields: Dict[str, Field]):
        self._fields = fields

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        Increments counts in the given ``counter`` for all of the vocabulary items in all of the
        ``Fields`` in this ``Instance``.
        """
        for _, field in self._fields.items():
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary):
        """
        Converts all ``UnindexedFields`` in this ``Instance`` to ``IndexedFields``, given the
        ``Vocabulary``.  This `mutates` the current object, it does not return a new ``Instance``.
        """
        for key, field in self._fields.items():
            if isinstance(field, UnindexedField):
                self._fields[key] = field.index(vocab)

    def get_padding_lengths(self) -> Dict[str, Dict[str, int]]:
        """
        Returns a dictionary of padding lengths, keyed field name.  Each ``Field`` returns a
        mapping from padding keys to actual lengths, and we just key that dictionary by field name.
        """
        lengths = {}
        for key, field in self._fields.items():
            lengths[key] = field.get_padding_lengths()
        return lengths

    def pad(self, padding_lengths: Dict[str, Dict[str, int]]) -> Dict[str, List[numpy.array]]:
        """
        Pads each ``Field`` in this instance to the lengths given in ``padding_lengths`` (which is
        keyed by field name, then by padding key, the same as the return value in
        :func:`get_padding_lengths`), returning a list of numpy arrays for each field.
        """
        arrays = {}
        for key, field in self._fields.items():
            field_lengths = padding_lengths.get(key, {})
            arrays[key] = field.pad(field_lengths)
        return arrays
