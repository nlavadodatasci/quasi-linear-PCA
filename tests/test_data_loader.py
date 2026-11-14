"""The chapter 4 data are package data; QLPCA_DATA can point to another copy."""
from importlib import resources

import pytest

from qlpca.datasets import load_st_gender, make_cylinder


def test_loads_from_package_data(monkeypatch):
    monkeypatch.delenv("QLPCA_DATA", raising=False)
    X, variables, countries = load_st_gender()
    assert X.shape == (15, 11) and len(variables) == 11 and len(countries) == 15


def test_qlpca_data_folder_is_used(monkeypatch, tmp_path):
    src = resources.files("qlpca") / "data" / "st_gender_eu15.csv"
    text = src.read_text(encoding="utf8").splitlines()
    (tmp_path / "st_gender_eu15.csv").write_text("\n".join(text[:4]) + "\n", encoding="utf8")
    monkeypatch.setenv("QLPCA_DATA", str(tmp_path))
    X, _, _ = load_st_gender()
    assert X.shape == (3, 11)


def test_missing_file_in_qlpca_data_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QLPCA_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="QLPCA_DATA"):
        load_st_gender()


def test_cylinder_needs_no_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("QLPCA_DATA", str(tmp_path))
    X, names = make_cylinder(n=20, random_state=0)
    assert X.shape == (20, 12) and len(names) == 12
