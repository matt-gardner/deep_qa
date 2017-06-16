from typing import Dict

from . import Field


class SequenceField(Field):
    """
    A ``SequenceField`` represents a sequence of things.  This class just adds a method onto
    ``Field``: :func:`sequence_length`.
    """
    def sequence_length(self, padding_lengths: Dict[str, int]) -> int:
        """
        Different ``SequenceFields`` might choose to use different keys to represent how many items
        are in their sequence.  This normalizes them into a consistent API, so that things like
        ``IndexField`` and ``TagField`` can just have one method to call.  The intent is that a
        class that needs to know sequence lengths can first call ``field.get_padding_lengths()``,
        then call ``field.sequence_length()`` with the result.
        """
        raise NotImplementedError
