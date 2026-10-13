import copy

import pytest

torch = pytest.importorskip("torch")

from pytest_torch import (
    ModelLoader,
    assert_generation_equal,
    assert_mask_invariant,
    assert_matches_reference,
    assert_padding_invariant,
)


@pytest.fixture(scope="module")
def tiny_bert_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.BertConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    path = tmp_path_factory.mktemp("tiny_bert")
    transformers.BertModel(config).save_pretrained(path)
    return str(path)


def _bert_batch():
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1, 64, (3, 6), generator=g)
    mask = torch.ones_like(ids)
    mask[1, 4:] = 0
    mask[2, 3:] = 0
    return {"input_ids": ids, "attention_mask": mask}


def _tiny_gpt2():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.GPT2Config(vocab_size=64, n_embd=16, n_layer=1, n_head=2, n_positions=64)
    return transformers.GPT2LMHeadModel(config).eval()


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #
def test_hf_loader_caches_and_sets_eval_mode(tiny_bert_dir):
    loader = ModelLoader()
    first = loader.hf(tiny_bert_dir)
    assert first is loader.hf(tiny_bert_dir)
    assert not first.training


def test_real_bert_is_padding_and_mask_invariant(tiny_bert_dir):
    model = ModelLoader().hf(tiny_bert_dir)
    batch = _bert_batch()
    assert_padding_invariant(model, batch, atol=1e-4)
    assert_mask_invariant(model, batch, vocab_size=64, atol=1e-4)


def test_bert_matches_copy_of_itself(tiny_bert_dir):
    model = ModelLoader().hf(tiny_bert_dir)
    assert_matches_reference(
        model, copy.deepcopy(model), _bert_batch(), select=lambda out: out.last_hidden_state
    )


def test_generation_equal_for_identical_models():
    model = _tiny_gpt2()
    ids = torch.randint(1, 64, (2, 5))
    assert_generation_equal(model, copy.deepcopy(model), ids, max_new_tokens=6, pad_token_id=0)


def test_generation_divergence_is_reported():
    model = _tiny_gpt2()
    other = copy.deepcopy(model)
    with torch.no_grad():
        embedding = other.transformer.wte.weight
        embedding.add_(torch.randn_like(embedding) * 5)
    ids = torch.randint(1, 64, (2, 5))
    with pytest.raises(AssertionError, match="Generated sequences differ"):
        assert_generation_equal(model, other, ids, max_new_tokens=6, pad_token_id=0)


# --------------------------------------------------------------------------- #
# timm / torchvision (random init: no downloads)
# --------------------------------------------------------------------------- #
def test_torchvision_loader_caches():
    pytest.importorskip("torchvision")
    loader = ModelLoader()
    model = loader.torchvision("resnet18")
    assert model is loader.torchvision("resnet18")
    assert not model.training


def test_timm_loader_caches():
    pytest.importorskip("timm")
    loader = ModelLoader()
    model = loader.timm("resnet18")
    assert model is loader.timm("resnet18")
    assert not model.training
