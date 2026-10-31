# Read vocabularies saved with an old Dill class.

import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import dill


def ignore_function(*args, **kwargs):
    return None


def vocabulary_container(*args, **kwargs):
    return SimpleNamespace


def load_type(name):
    return {"type": type, "object": object, "CodeType": ignore_function}[name]


class DictionaryStateReader(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("dill._dill", "_create_type"): vocabulary_container,
            ("dill._dill", "_load_type"): load_type,
            ("dill._dill", "_create_function"): ignore_function,
            ("__builtin__", "__main__"): {},
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(
                "Unexpected vocabulary object: " + module + "." + name
            )
        return allowed[(module, name)]


def canonical(vocabulary):
    return {
        name: {
            "idx2word": sorted(item.idx2word.items()),
            "word2idx": sorted(item.word2idx.items()),
        }
        for name, item in sorted(vocabulary.items())
    }


def convert(source, target):
    with Path(source).open("rb") as stream:
        vocabulary = DictionaryStateReader(stream).load()
    assert set(vocabulary) == {"diag_voc", "pro_voc", "med_voc"}
    for name, item in vocabulary.items():
        assert set(vars(item)) == {"idx2word", "word2idx"}
        assert set(item.idx2word) == set(range(len(item.idx2word)))
        assert len(item.idx2word) == len(item.word2idx)
        assert all(
            item.word2idx[word] == index for index, word in item.idx2word.items()
        )
    before = json.dumps(
        canonical(vocabulary), ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    with Path(target).open("wb") as stream:
        pickle.dump(vocabulary, stream, protocol=4)
    with Path(target).open("rb") as stream:
        reread = dill.load(stream)
    after = json.dumps(canonical(reread), ensure_ascii=False, sort_keys=True).encode(
        "utf-8"
    )
    assert before == after
    return vocabulary, {
        "canonical_sha256": hashlib.sha256(before).hexdigest(),
        "counts": {name: len(item.idx2word) for name, item in vocabulary.items()},
        "scope": "Original index-to-code and code-to-index dictionaries retained; obsolete serialized class methods are unnecessary for author training.",
    }
