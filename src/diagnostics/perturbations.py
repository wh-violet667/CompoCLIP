"""可控组合变换：所有函数保持词袋不变（只改词序/句法角色），用于 0-B 词序探针。

设计原则：
- 同一组词、只改顺序或句法位置，排除“换词”带来的语义先验；
- 变换失败（找不到主语/宾语、没有两个名词等）返回 None，由调用方丢弃并计数；
- 随机变换固定 seed，保证可复现。

spaCy 模型：en_core_web_sm；首次调用时懒加载。
"""
from __future__ import annotations

import random
import re
from functools import lru_cache

SUBJ_DEPS = {"nsubj", "nsubjpass"}
OBJ_DEPS = {"dobj", "obj", "pobj"}
NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
ADJ_TAGS = {"JJ", "JJR", "JJS"}


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    return spacy.load("en_core_web_sm")


def _swap(doc, i: int, j: int) -> str:
    tokens = [t.text for t in doc]
    tokens[i], tokens[j] = tokens[j], tokens[i]
    return " ".join(tokens)


def _changed(original: str, new: str) -> str | None:
    return new if new.strip().lower() != original.strip().lower() else None


def swap_subj_obj(caption: str) -> str | None:
    """交换主语与宾语（同一组词，句法角色相反）。"""
    doc = _nlp()(caption)
    subj = [t.i for t in doc if t.dep_ in SUBJ_DEPS]
    obj = [t.i for t in doc if t.dep_ in OBJ_DEPS and t.i not in subj]
    if not subj or not obj:
        return None
    return _changed(caption, _swap(doc, subj[0], obj[0]))


def swap_two_nouns(caption: str, rng: random.Random | None = None) -> str | None:
    """交换两个名词（优先跨句法成分，制造词序/角色冲突）。"""
    doc = _nlp()(caption)
    nouns = [t.i for t in doc if t.tag_ in NOUN_TAGS]
    if len(nouns) < 2:
        return None
    rng = rng or random.Random(0)
    i, j = rng.sample(nouns, 2)
    return _changed(caption, _swap(doc, i, j))


def swap_two_adjectives(caption: str, rng: random.Random | None = None) -> str | None:
    """交换两个形容词（属性错配，词袋不变）。"""
    doc = _nlp()(caption)
    adjs = [t.i for t in doc if t.tag_ in ADJ_TAGS]
    if len(adjs) < 2:
        return None
    rng = rng or random.Random(0)
    i, j = rng.sample(adjs, 2)
    return _changed(caption, _swap(doc, i, j))


def shuffle_all_words(caption: str, rng: random.Random | None = None) -> str | None:
    """完全打乱词序：词袋基线；若模型仍高分，说明主要靠词袋匹配。"""
    words = [t.text for t in _nlp()(caption)]
    if len(words) < 3:
        return None
    rng = rng or random.Random(0)
    shuffled = words[:]
    for _ in range(10):
        rng.shuffle(shuffled)
        if shuffled != words:
            break
    return _changed(caption, " ".join(shuffled))


def shuffle_nouns_and_adj(caption: str, rng: random.Random | None = None) -> str | None:
    """只打乱名词与形容词（保持虚词/动词骨架），对应 ARO Order 的扰动。"""
    doc = _nlp()(caption)
    tokens = [t.text for t in doc]
    idx = [i for i, t in enumerate(doc) if t.tag_ in NOUN_TAGS | ADJ_TAGS]
    if len(idx) < 2:
        return None
    rng = rng or random.Random(0)
    vals = [tokens[i] for i in idx]
    rng.shuffle(vals)
    for i, v in zip(idx, vals):
        tokens[i] = v
    return _changed(caption, " ".join(tokens))


def shuffle_all_but_nouns_and_adj(caption: str, rng: random.Random | None = None) -> str | None:
    """打乱除名词/形容词外的所有词（保持实体词位置）。"""
    doc = _nlp()(caption)
    tokens = [t.text for t in doc]
    idx = [i for i, t in enumerate(doc) if t.tag_ not in NOUN_TAGS | ADJ_TAGS]
    if len(idx) < 2:
        return None
    rng = rng or random.Random(0)
    vals = [tokens[i] for i in idx]
    rng.shuffle(vals)
    for i, v in zip(idx, vals):
        tokens[i] = v
    return _changed(caption, " ".join(tokens))


def _trigrams(words: list[str]) -> list[list[str]]:
    return [words[i:i + 3] for i in range(0, len(words), 3)]


def shuffle_within_trigrams(caption: str, rng: random.Random | None = None) -> str | None:
    rng = rng or random.Random(0)
    groups = _trigrams([t.text for t in _nlp()(caption)])
    for g in groups:
        rng.shuffle(g)
    return _changed(caption, " ".join(" ".join(g) for g in groups))


def shuffle_trigrams(caption: str, rng: random.Random | None = None) -> str | None:
    rng = rng or random.Random(0)
    groups = _trigrams([t.text for t in _nlp()(caption)])
    rng.shuffle(groups)
    return _changed(caption, " ".join(" ".join(g) for g in groups))


REL_PATTERN = re.compile(
    r"^(?P<a>.+?)\s+(?:is\s+)?(?P<rel>to the (?:left|right|front|back) of|"
    r"above|below|under|on|in|near|behind|in front of)\s+(?P<b>.+)$",
    re.IGNORECASE,
)


def reverse_relation(caption: str) -> str | None:
    """关系方向反转：'A to the left of B' -> 'B to the left of A'（词袋不变）。"""
    m = REL_PATTERN.match(caption.strip().rstrip("."))
    if not m:
        return None
    a, rel, b = m.group("a").strip(), m.group("rel"), m.group("b").strip()
    if not a or not b:
        return None
    return _changed(caption, f"{b} {rel} {a}")


PROBES = {
    "subj_obj_swap": swap_subj_obj,
    "noun_swap": swap_two_nouns,
    "adj_swap": swap_two_adjectives,
    "word_shuffle": shuffle_all_words,
    "trigram_shuffle": shuffle_within_trigrams,
    "relation_reverse": reverse_relation,
}
