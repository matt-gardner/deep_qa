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
    def needs_indexing(self):
        """
        If there are strings in this ``Field`` that need to be indexed, you must return ``True``
        here; else return ``False``.
        """
        raise NotImplementedError

    # pylint: disable=no-self-use,unused-argument
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.

        Only called if ``self.needs_indexing()`` returns ``True``.  Because of this, we raise a
        ``RuntimeError`` here instead of a ``NotImplementedError``, because fields that are already
        indexed don't need to implement this.
        """
        raise RuntimeError("You need to implement this method, or return False in needs_indexing")

    def index(self, vocab: Vocabulary):
        """
        Given a :class:`Vocabulary`, converts all strings in this field into (typically) integers.
        This `modifies` the ``Field`` object, it does not return anything.

        Only called if ``self.needs_indexing()`` returns ``True``.  Because of this, we raise a
        ``RuntimeError`` here instead of a ``NotImplementedError``, because fields that are already
        indexed don't need to implement this.
        """
        raise RuntimeError("You need to implement this method, or return False in needs_indexing")
    # pylint: enable=no-self-use,unused-argument

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
