"""Pins the behaviour of the changes that make re-matching a cached scene ~6x cheaper (432 -> 70 ms measured on real
scenes, identical results): batched per-scene index search, the cached merged link index, memoized URL normalization."""
import numpy as np

import matching
from matching import IndexQueryResult, MatchingConfig, prefetched_searches, query_index
from scene_matcher import _recognizer_link_index
from stashbox_utils import normalize_url_for_compare


class _Batch:
    def __init__(self, keys, distances, counts):
        self.keys, self.distances, self.counts = keys, distances, counts


class FakeIndex:
    """search() on a 1D vector -> one result; on a 2D matrix -> a batch. Deterministic per query."""
    def __init__(self, size=10):
        self.size = size
        self.single_calls = 0
        self.batch_calls = 0

    def __len__(self):
        return self.size

    @staticmethod
    def _one(vec, k):
        # neighbour ids / distances derived from the vector so each query has its own answer
        base = int(abs(vec.sum()) * 1000) % 1000
        keys = np.array([base + j for j in range(k)], dtype=np.uint64)
        dist = np.array([0.01 * (j + 1) for j in range(k)], dtype=np.float32)
        return keys, dist

    def search(self, vectors, k, threads=None):
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            self.single_calls += 1
            keys, dist = self._one(vectors, k)
            return _Batch(keys, dist, k)
        self.batch_calls += 1
        out = [self._one(v, k) for v in vectors]
        return _Batch(np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.full(len(vectors), k))


def _vec(seed):
    return np.random.default_rng(seed).standard_normal(8).astype(np.float32)


CFG = MatchingConfig(query_k=5)


def test_prefetched_results_equal_single_queries_and_use_one_batch_call():
    index = FakeIndex()
    embs = [_vec(i) for i in range(6)]
    singles = [query_index(e, index, CFG) for e in embs]
    assert index.single_calls == 6
    with prefetched_searches([(index, embs)], CFG.query_k):
        batched = [query_index(e, index, CFG) for e in embs]
    assert index.batch_calls == 1 and index.single_calls == 6      # no extra single queries inside the block
    for s, b in zip(singles, batched):
        assert list(s.neighbors) == list(b.neighbors) and list(s.distances) == list(b.distances)


def test_unprefetched_embedding_and_other_index_fall_through_to_single_search():
    index, other = FakeIndex(), FakeIndex()
    with prefetched_searches([(index, [_vec(1)])], CFG.query_k):
        query_index(_vec(2), index, CFG)       # same index, embedding not prefetched
        query_index(_vec(1), other, CFG)       # prefetched embedding but another index
    assert index.single_calls == 1 and other.single_calls == 1


def test_prefetch_is_scoped_to_the_block():
    index = FakeIndex()
    e = _vec(3)
    with prefetched_searches([(index, [e])], CFG.query_k):
        pass
    query_index(e, index, CFG)
    assert index.single_calls == 1


def test_a_failing_or_empty_index_does_not_break_matching():
    class Boom(FakeIndex):
        def search(self, vectors, k, threads=None):
            if np.asarray(vectors).ndim == 2:
                raise RuntimeError("batch not supported")
            return super().search(vectors, k)

    boom, empty = Boom(), FakeIndex(size=0)
    e = _vec(4)
    with prefetched_searches([(boom, [e]), (empty, [e]), (None, [e])], CFG.query_k):
        assert isinstance(query_index(e, boom, CFG), IndexQueryResult)
    assert empty.batch_calls == 0


class _Rec:
    def __init__(self, plink, local):
        self.performer_link_index = plink
        self.local_catalogue_link_index = local


def test_merged_link_index_is_computed_once_per_pair_of_source_dicts():
    plink, local = {"a": ["b"], "b": ["a"]}, {"a": ["local:1"], "local:1": ["a"]}
    rec = _Rec(plink, local)
    first = _recognizer_link_index(rec)
    assert sorted(first["a"]) == ["b", "local:1"]
    assert _recognizer_link_index(rec) is first              # served from the cache


def test_merged_link_index_follows_a_replaced_source_dict():
    rec = _Rec({"a": ["b"], "b": ["a"]}, {})
    first = _recognizer_link_index(rec)
    rec.local_catalogue_link_index = {"b": ["local:9"], "local:9": ["b"]}   # local index reloaded -> a NEW dict
    second = _recognizer_link_index(rec)
    assert second is not first and "local:9" in second["b"]


def test_url_normalization_is_unchanged_by_memoization():
    for raw, want in [("https://www.OnlyFans.com/x/", "onlyfans.com/x"), ("http://a.b/c", "a.b/c"),
                      ("www.x.com", "x.com"), ("  ", ""), (None, ""), ("", "")]:
        assert normalize_url_for_compare(raw) == want
        assert normalize_url_for_compare(raw) == want          # second call hits the cache


def test_native_frame_size_is_only_fetched_for_rotation_corrected_video_faces():
    from types import SimpleNamespace
    from identification_router import _needs_native_frame_size

    def res(bbox):
        return SimpleNamespace(face=SimpleNamespace(bbox=bbox))

    plain = [(0, res({"x": 1, "y": 1, "w": 5, "h": 5, "rotation_applied": 0.0})), (3, res({"x": 1, "y": 1, "w": 5, "h": 5}))]
    assert _needs_native_frame_size(plain) is False
    assert _needs_native_frame_size([]) is False
    assert _needs_native_frame_size(plain + [(5, res({"rotation_applied": 90.0}))]) is True
    # a rotated SPRITE face (negative frame_index) never uses the native size
    assert _needs_native_frame_size([(-2, res({"rotation_applied": 90.0}))]) is False
