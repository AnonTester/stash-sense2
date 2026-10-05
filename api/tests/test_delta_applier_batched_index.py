"""delta_applier._BatchedIndex -- queues usearch add() into multi-threaded batches (one-at-a-time adds cost ~1.9 ms each
against the ~1M-vector index, 84 s of a 94 s apply for a 44k-face delta). Tested against a fake index so the semantics
apply_delta_db relies on are pinned without a real usearch index: queued keys count as present, a removed queued key is
never added, everything is flushed before save, and flushes stay small (they hold the GIL)."""
import numpy as np
import pytest

from delta_applier import _BatchedIndex

pytestmark = pytest.mark.heavy  # delta_applier imports usearch at module level


class FakeIndex:
    def __init__(self, keys=()):
        self.keys = set(keys)
        self.add_calls = []       # (keys, vectors, threads) per batched call
        self.saved = None

    def __contains__(self, key):
        return int(key) in self.keys

    def add(self, keys, vectors, threads=None):
        assert len(keys) == len(vectors)
        self.add_calls.append((list(map(int, keys)), vectors.copy(), threads))
        for k in keys:
            assert int(k) not in self.keys, "batched add of a key already in the index"
            self.keys.add(int(k))

    def remove(self, key):
        self.keys.discard(int(key))

    def save(self, path):
        self.saved = (path, set(self.keys))


def _vec(seed):
    v = np.random.default_rng(seed).standard_normal(8).astype(np.float32)
    return v / np.linalg.norm(v)


def test_queued_keys_count_as_present_and_nothing_is_added_until_flush():
    fake = FakeIndex()
    idx = _BatchedIndex(fake)
    idx.add(7, _vec(1))
    assert 7 in idx and 7 not in fake and fake.add_calls == []
    idx.flush()
    assert 7 in fake and len(fake.add_calls) == 1


def test_remove_then_add_replaces_an_existing_key():
    fake = FakeIndex(keys=[5])
    idx = _BatchedIndex(fake)
    assert 5 in idx
    idx.remove(5)            # what _upsert_face does for an existing embedding_index
    idx.add(5, _vec(2))
    idx.flush()
    assert fake.add_calls[0][0] == [5] and 5 in fake


def test_removing_a_queued_key_drops_it_without_touching_the_index():
    fake = FakeIndex()
    idx = _BatchedIndex(fake)
    idx.add(9, _vec(3))
    idx.remove(9)
    idx.flush()
    assert 9 not in idx and fake.add_calls == []


def test_adding_the_same_key_twice_keeps_the_last_vector():
    fake = FakeIndex()
    idx = _BatchedIndex(fake)
    idx.add(4, _vec(10))
    idx.add(4, _vec(11))
    idx.flush()
    keys, vectors, _ = fake.add_calls[0]
    assert keys == [4]
    assert np.allclose(vectors[0], _vec(11))


def test_save_flushes_first():
    fake = FakeIndex()
    idx = _BatchedIndex(fake)
    idx.add(1, _vec(1))
    idx.add(2, _vec(2))
    idx.save("/tmp/x.usearch")
    assert fake.saved == ("/tmp/x.usearch", {1, 2})


def test_batches_are_bounded_and_thread_count_is_capped():
    fake = FakeIndex()
    idx = _BatchedIndex(fake)
    n = _BatchedIndex.MAX_PENDING * 2 + 10
    for k in range(n):
        idx.add(k, _vec(k % 5))
    idx.flush()
    assert max(len(c[0]) for c in fake.add_calls) <= _BatchedIndex.MAX_PENDING
    assert sum(len(c[0]) for c in fake.add_calls) == n
    assert all(1 <= c[2] <= _BatchedIndex.MAX_THREADS for c in fake.add_calls)
