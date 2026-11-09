"""loader 分组与限流单测（基于临时 JSONL，不依赖真实数据集）。"""
import json
import warnings

import loaders


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def test_group_by_image_keeps_multi_caption(tmp_path):
    rows = [
        {"image": "a.jpg", "caption": f"cap {i}", "image_id": "1", "caption_id": str(i)}
        for i in range(5)
    ] + [
        {"image": "b.jpg", "caption": f"other {i}", "image_id": "2", "caption_id": str(i)}
        for i in range(5)
    ]
    _write(tmp_path / "flickr30k/splits/test.jsonl", rows)

    groups = loaders.load_flickr30k(tmp_path, split="test")
    assert len(groups) == 2
    assert all(len(g.captions) == 5 for g in groups)
    assert groups[0].image.name == "a.jpg"
    assert groups[0].captions[0] == "cap 0"


def test_limit_counts_images_not_captions(tmp_path):
    rows = [
        {"image": f"{img}.jpg", "caption": f"c{i}", "image_id": str(img), "caption_id": str(i)}
        for img in range(4) for i in range(5)
    ]
    _write(tmp_path / "flickr30k/splits/test.jsonl", rows)

    groups = loaders.load_flickr30k(tmp_path, split="test", limit=2)
    assert len(groups) == 2


def test_warns_when_caption_count_is_not_five(tmp_path):
    rows = [{"image": "a.jpg", "caption": "only one", "image_id": "1", "caption_id": "0"}]
    _write(tmp_path / "flickr30k/splits/test.jsonl", rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaders.load_flickr30k(tmp_path, split="test")
    assert any("不是 5" in str(w.message) for w in caught)


def test_missing_split_has_actionable_error(tmp_path):
    try:
        loaders.load_flickr30k(tmp_path, split="test")
    except FileNotFoundError as e:
        assert "prepare_flickr30k" in str(e)
    else:
        raise AssertionError("缺少 split 时应报错")


def test_load_pairwise(tmp_path):
    path = tmp_path / "golden" / "aro.jsonl"
    _write(path, [
        {"image": "img.jpg", "positive": "dog left of cat", "negative": "cat left of dog",
         "subset": "vg_relation"},
    ])
    items = loaders.load_pairwise(path)
    assert len(items) == 1
    assert items[0].subset == "vg_relation"
    assert items[0].positive != items[0].negative
