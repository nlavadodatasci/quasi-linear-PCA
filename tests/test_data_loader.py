"""The chapter 4 loader needs the repository data folder; elsewhere it must fail clearly."""
import pytest

from qlpca.datasets import load_st_gender, make_cylinder


def test_missing_data_gives_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("QLPCA_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="QLPCA_DATA") as err:
        load_st_gender()
    assert "github.com/nlavadodatasci/quasi-linear-PCA" in str(err.value)
    assert "make_cylinder" in str(err.value)


def test_cylinder_needs_no_data_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("QLPCA_DATA", str(tmp_path))
    X, names = make_cylinder(n=20, random_state=0)
    assert X.shape == (20, 12) and len(names) == 12
