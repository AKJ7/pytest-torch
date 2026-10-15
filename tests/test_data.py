import pytest

torch = pytest.importorskip("torch")

from pytest_torch.data import BatchFactory


@pytest.mark.parametrize("padding", ["right", "left"])
def test_tokens_mix_padded_and_full_rows(make_batch, torch_device, padding):
    batch = make_batch.tokens(6, 16, vocab_size=50, padding=padding, labels="lm")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    assert ids.shape == mask.shape == (6, 16) and ids.dtype == mask.dtype == torch.long
    assert ids.device.type == torch.device(torch_device).type
    lengths = mask.sum(1)
    assert lengths[0] == 16 and lengths[-1] == 8 and bool((lengths >= 8).all())
    real = mask.bool()
    assert bool((ids[~real] == 0).all()) and bool((ids[real] != 0).all())  # pad id only on padding
    assert bool((ids < 50).all())
    if padding == "right":
        assert bool(mask[:, 0].all())  # every row starts with a real token
    else:
        assert bool(mask[:, -1].all())
    assert bool((batch["labels"][~real] == -100).all()) and torch.equal(batch["labels"][real], ids[real])


def test_tokens_options(make_batch):
    full = make_batch.tokens(3, 8, padding=None, labels=5, token_type_ids=True, pad_token_id=7)
    assert bool(full["attention_mask"].all()) and bool((full["input_ids"] != 7).all())
    assert full["labels"].shape == (3,) and bool((full["labels"] < 5).all())
    assert bool((full["token_type_ids"] == 0).all())
    one = make_batch.tokens(1, 8)  # a single row has the full length
    assert one["attention_mask"].sum() == 8
    with pytest.raises(ValueError, match="padding"):
        make_batch.tokens(2, 8, padding="middle")
    with pytest.raises(ValueError, match="min_length"):
        make_batch.tokens(2, 8, min_length=9)
    with pytest.raises(ValueError, match="pad_token_id"):
        make_batch.tokens(2, 8, vocab_size=10, pad_token_id=10)


def test_images(make_batch):
    images = make_batch.images(2, 3, 32)
    assert images.shape == (2, 3, 32, 32) and images.dtype == torch.float32
    raw, labels = make_batch.images(4, 3, 16, 24, uint8=True, layout="nhwc", labels=10)
    assert raw.shape == (4, 16, 24, 3) and raw.dtype == torch.uint8 and int(raw.max()) <= 255
    assert labels.shape == (4,) and bool((labels < 10).all())
    assert make_batch.images(1, 1, 8, dtype=torch.float64).dtype == torch.float64


def test_loader_is_seeded(make_batch, torch_seed):
    data = (torch.arange(10), torch.arange(10) * 2)
    order = [x.tolist() for x, _ in make_batch.loader(data, batch_size=4, shuffle=True)]
    again = BatchFactory(make_batch.make_tensor, seed=torch_seed)
    assert order == [x.tolist() for x, _ in again.loader(data, batch_size=4, shuffle=True)]
    assert sorted(i for batch in order for i in batch) == list(range(10)) and order != [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    rows = list(make_batch.loader({"x": torch.zeros(5, 2), "y": torch.ones(5)}, batch_size=5))
    assert rows[0]["x"].shape == (5, 2) and rows[0]["y"].shape == (5,)
    with pytest.raises(ValueError, match="different lengths"):
        make_batch.loader({"x": torch.zeros(5), "y": torch.zeros(4)})


def test_batches_are_reproducible_and_recorded(pytester):
    pytester.makepyfile(
        """
        import torch

        def test_tokens(make_batch):
            batch = make_batch.tokens(4, 8)
            print("IDS", batch["input_ids"].sum().item(), batch["attention_mask"].sum().item())
            assert False
        """
    )
    first = pytester.runpytest("-p", "no:cacheprovider", "--torch-device=cpu", "--torch-repro-dir=repro")
    first.assert_outcomes(failed=1)
    line = next(out for out in first.outlines if out.startswith("IDS"))
    bundle = next((pytester.path / "repro").glob("*"))
    other_seed = pytester.runpytest("-p", "no:cacheprovider", "--torch-device=cpu", "--torch-seed=99")
    assert line not in other_seed.outlines
    replay = pytester.runpytest("-p", "no:cacheprovider", "--torch-device=cpu", "--torch-seed=99",
                                f"--torch-repro={bundle}")
    assert line in replay.outlines  # replayed bit for bit from the bundle
