from typing import Dict, List

import numpy

from ..vocabulary import Vocabulary

class Field:
    """
    A ``Field`` is some piece of a data instance that ends up as an array in a model (either as an
    input or an output).  Data instances are just collections of fields.


    Fields go through up to two steps of processing: (1) tokenized fields are converted into token ids,
    (2) fields containing token ids (or any other numeric data) are padded (if necessary) and converted
    into data arrays.  The ``Field`` object has methods to say which state the field is in, and to get
    it from one state to the other.  If ``Field.needs_indexing()``, we will compute a vocabulary and
    pass it to the field to use for indexing.  Once all fields are indexed, we will determine padding
    lengths, then intelligently batch together instances and pad them into actual arrays.
    """
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the ``Field`` object, it does not return anything.

        If your ``Field`` does not have any strings that need to be converted into indices, you do
        not need to implement this method.
        """
        pass

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like {'num_tokens': 13}.

        This is always called after :func:`index`.
        """
        raise NotImplementedError

    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        numpy array of the correct shape.  This actually returns a list instead of a single array,
        in case there are several related arrays for this field (e.g., a ``TextField`` might have a
        word array and a characters-per-word array).
        """
        raise NotImplementedError

    def empty_field(self) -> 'Field':
        """
        So that ``ListField`` can pad the number of fields in a list (e.g., the number of answer
        option ``TextFields``), we need a representation of an empty field of each type.  This
        returns that.  This will only ever be called when we're to the point of calling
        :func:`pad`, so you don't need to worry about ``get_padding_lengths``,
        ``count_vocab_items``, etc., being called on this empty field.

        We make this an instance method instead of a static method so that if there is any state
        in the Field, we can copy it over (e.g., the token indexers in ``TextField``).
        """
        raise NotImplementedError
