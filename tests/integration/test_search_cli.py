from __future__ import annotations

import argparse

import pytest

import search
from tests.conftest import FakeEmbedder


@pytest.mark.integration
def test_search_cli_text_query_returns_expected_family_vehicle(
    monkeypatch,
    capsys,
    sample_dataset_root,
    sample_data_dir,
    sample_vectors,
    sample_metadata_records,
) -> None:
    monkeypatch.setattr(
        search,
        "parse_args",
        lambda: argparse.Namespace(
            text="family suv cargo",
            image=None,
            pdf=None,
            finance_file=None,
            modality="all",
            top_k=3,
            data_dir=str(sample_data_dir),
            dataset_root=str(sample_dataset_root),
            verbose=False,
        ),
    )
    monkeypatch.setattr(search, "GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr(search, "load_index_artifacts", lambda settings: (sample_vectors, sample_metadata_records))
    monkeypatch.setattr(search, "configure_logging", lambda verbose=False: None)

    search.main()
    output = capsys.readouterr().out

    assert "2026 Acme Family Cruiser" in output
    assert "Search mode: all" in output


@pytest.mark.integration
def test_search_cli_finance_filter_prefers_finance_rows(
    monkeypatch,
    capsys,
    sample_dataset_root,
    sample_data_dir,
    sample_vectors,
    sample_metadata_records,
) -> None:
    monkeypatch.setattr(
        search,
        "parse_args",
        lambda: argparse.Namespace(
            text="commercial bonus cash truck",
            image=None,
            pdf=None,
            finance_file=None,
            modality="finance",
            top_k=2,
            data_dir=str(sample_data_dir),
            dataset_root=str(sample_dataset_root),
            verbose=False,
        ),
    )
    monkeypatch.setattr(search, "GeminiEmbedder", FakeEmbedder)
    monkeypatch.setattr(search, "load_index_artifacts", lambda settings: (sample_vectors, sample_metadata_records))
    monkeypatch.setattr(search, "configure_logging", lambda verbose=False: None)

    search.main()
    output = capsys.readouterr().out

    assert "Search mode: finance" in output
    assert "2026 Acme Payload Pro" in output
