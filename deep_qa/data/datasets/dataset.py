import logging
from collections import defaultdict
from typing import Dict, List

import numpy
import tqdm

from ...common.util import add_noise_to_dict_values
from .. import Vocabulary
from .. import Instance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dataset:
    """
    A collection of Instances.

    This base class has general methods that apply to all collections of Instances.  That basically
    is just methods that operate on sets, like merging and truncating.
    """
    def __init__(self, instances: List[Instance]):
        """
        A Dataset just takes a list of instances in its constructor.  It's important that all
        subclasses have an identical constructor to this (though possibly with different Instance
        types).  If you change the constructor, you also have to override all methods in this base
        class that call the constructor, such as `merge()` and `truncate()`.
        """
        self.instances = instances

    def truncate(self, max_instances: int):
        """
        If there are more instances than `max_instances` in this dataset, returns a new dataset
        with a random subset of size `max_instances`.  If there are fewer than `max_instances`
        already, we just return self.
        """
        if len(self.instances) <= max_instances:
            return self
        new_instances = [i for i in self.instances]
        return self.__class__(new_instances[:max_instances])

    def index_instances(self, vocab: Vocabulary):
        """
        Converts all ``UnindexedFields`` in all ``Instances`` in this ``Dataset`` into
        ``IndexedFields``.  This modifies the current object, it does not return a new object.
        """
        for instance in tqdm.tqdm(self.instances):
            instance.index_fields(vocab)

    def sort_by_padding(self, sorting_keys: List[(str, str)], padding_noise: float=0.0):
        """
        Sorts the ``Instances`` in this ``Dataset`` by their padding lengths, using the keys in
        ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
        ``(field_name, padding_key)`` tuples.
        """
        instances_with_lengths = []
        for instance in self.instances:
            padding_lengths = instance.get_padding_lengths()
            if padding_noise > 0.0:
                noisy_lengths = {}
                for field_name, field_lengths in padding_lengths:
                    noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
                padding_lengths = noisy_lengths
            instance_with_lengths = [padding_lengths[field_name][padding_key]
                                     for (field_name, padding_key) in sorting_keys] + [instance]
            instances_with_lengths.append(instance_with_lengths)
        instances_with_lengths.sort(key=lambda x: x[:-1])
        self.instances = [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]

    def padding_lengths(self):
        padding_lengths = {}
        all_instance_lengths = [instance.get_padding_lengths() for instance in self.instances]
        if not all_instance_lengths:
            return padding_lengths
        all_field_lengths = defaultdict(list)
        for instance_lengths in all_instance_lengths:
            for field_name, instance_field_lengths in instance_lengths.items():
                all_field_lengths[field_name].append(instance_field_lengths)
        for field_name, field_lengths in all_field_lengths.items():
            for padding_key in field_lengths[0].keys():
                max_value = max(x[padding_key] if padding_key in x else 0 for x in field_lengths)
                padding_lengths[field_name][padding_key] = max_value
        return padding_lengths

    def as_arrays(self,
                  padding_lengths: Dict[str, Dict[str, int]]=None,
                  verbose: bool=True) -> Dict[str, List[numpy.array]]:
        """
        This method converts this ``Dataset`` into a set of numpy arrays that can be passed through
        a model.  In order for the numpy arrays to be valid arrays, all ``Instances`` in this
        dataset need to be padded to the same lengths wherever padding is necessary, so we do that
        first, then we combine all of the arrays for each field in each instance into a set of
        batched arrays for each field.

        Parameters
        ----------
        padding_lengths : ``Dict[str, Dict[str, int]]``
            If a key is present in this dictionary with a non-``None`` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length if you want to throw out long sequences.

            Entries in this dictionary are keyed first by field name (e.g., "question"), then by
            padding key (e.g., "num_tokens").
        verbose : ``bool``, optional (default=``True``)
            Should we output logging information when we're doing this padding?  If the dataset is
            large, this is nice to have, because padding a large dataset could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious.

        Returns
        -------
        data_arrays : ``Dict[str, List[numpy.array]]``
            A dictionary of data arrays, keyed by field name, suitable for passing as input to a
            model.  This is a `batch` of instances, so, e.g., if the instances have a "question"
            field and an "answer" field, the "question" fields for all of the instances will be
            grouped together into a single array, and the "answer" fields for all instances will be
            similarly grouped in a parallel set of arrays, for batched computation.
        """
        if padding_lengths is None:
            padding_lengths = defaultdict(dict)
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular field and padding key.  If we were, we use that
        # instead of the instance-based one.
        if verbose:
            logger.info("Padding dataset of size %d to lengths %s", len(self.instances), str(padding_lengths))
            logger.info("Getting max lengths from instances")
        instance_padding_lengths = self.padding_lengths()
        if verbose:
            logger.info("Instance max lengths: %s", str(instance_padding_lengths))
        lengths_to_use = defaultdict(dict)
        for field_name, instance_field_lengths in instance_padding_lengths.items():
            for padding_key in instance_field_lengths:
                if padding_lengths[field_name].get(padding_key) is not None:
                    lengths_to_use[field_name][padding_key] = padding_lengths[padding_key]
                else:
                    lengths_to_use[field_name][padding_key] = instance_field_lengths[padding_key]

        # Now we actually pad the instances to numpy arrays.
        field_arrays = defaultdict(list)
        if verbose:
            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))
            for instance in tqdm.tqdm(self.instances):
                for field, arrays in instance.pad(lengths_to_use):
                    field_arrays[field].append(arrays)
        else:
            for instance in self.instances:
                for field, arrays in instance.pad(lengths_to_use):
                    field_arrays[field].append(arrays)

        # Finally, we combine the arrays that we got for each instance into one big array (or set
        # of arrays) per field.
        for field_name, field_array_list in field_arrays.items():
            if isinstance(field_array_list[0], [list, tuple]):
                field_arrays[field_name] = [numpy.asarray(x) for x in zip(*field_array_list)]
            else:
                field_arrays[field_name] = numpy.asarray(field_array_list)
        return field_arrays
