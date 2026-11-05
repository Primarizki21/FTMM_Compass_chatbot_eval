"""Memastikan bert-score cocok dengan transformers terpasang, memakai model kecil acak tanpa unduhan."""

import pytest


def test_bert_score_runs_with_installed_transformers(tmp_path):
    pytest.importorskip("bert_score")
    from bert_score import score
    from transformers import BertConfig, BertModel, BertTokenizer

    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "mata", "kuliah", "statistika", "dasar", "kucing", "hitam"]
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("\n".join(words), encoding="utf-8")
    BertTokenizer(str(vocab), model_max_length=512).save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=len(words), hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32
    )
    BertModel(config).save_pretrained(tmp_path)

    _, _, f1 = score(
        ["mata kuliah statistika", "kucing hitam", ""],
        ["mata kuliah statistika", "mata kuliah dasar", "mata kuliah"],
        model_type=str(tmp_path),
        num_layers=1,
        device="cpu",
        verbose=False,
    )

    assert f1[0].item() == pytest.approx(1.0, abs=1e-3)
    assert f1[2].item() == 0.0
