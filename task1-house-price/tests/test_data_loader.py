"""
Unit tests for data_loader module.

Run with: python -m pytest tests/test_data_loader.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import DataLoader, BASELINE_FEATURES, TARGET_COLUMN, ID_COLUMN


# Module-level fixtures for reuse across test classes
@pytest.fixture
def sample_train_csv():
    """Create a temporary train.csv for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data = {
            "Id": [1, 2, 3, 4, 5],
            "GrLivArea": [1500, 2000, 1800, 2200, 1600],
            "BedroomAbvGr": [3, 4, 3, 4, 3],
            "FullBath": [2, 3, 2, 3, 2],
            "SalePrice": [200000, 300000, 250000, 350000, 220000],
            "LotArea": [8000, 10000, 9000, 12000, 8500],
            "OverallQual": [7, 8, 7, 9, 6]
        }
        pd.DataFrame(data).to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_test_csv():
    """Create a temporary test.csv for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data = {
            "Id": [6, 7, 8],
            "GrLivArea": [1700, 2100, 1900],
            "BedroomAbvGr": [3, 4, 3],
            "FullBath": [2, 3, 2],
            "LotArea": [8500, 11000, 9500],
            "OverallQual": [7, 8, 7]
        }
        pd.DataFrame(data).to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_desc_txt():
    """Create a temporary data_description.txt."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("GrLivArea: Above ground living area\n")
        f.write("SalePrice: Sale price\n")
        f.flush()  # Ensure content is written
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def sample_submission_csv():
    """Create a temporary sample_submission.csv."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data = {"Id": [6, 7, 8], "SalePrice": [0, 0, 0]}
        pd.DataFrame(data).to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)

@pytest.fixture
def temp_data_dir(sample_train_csv, sample_test_csv, sample_desc_txt, sample_submission_csv):
    """Create a temporary data directory with all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy files to data_dir
        train_dest = Path(tmpdir) / "train.csv"
        test_dest = Path(tmpdir) / "test.csv"
        desc_dest = Path(tmpdir) / "data_description.txt"
        sub_dest = Path(tmpdir) / "sample_submission.csv"

        shutil.copy(sample_train_csv, train_dest)
        shutil.copy(sample_test_csv, test_dest)
        shutil.copy(sample_desc_txt, desc_dest)
        shutil.copy(sample_submission_csv, sub_dest)

        yield tmpdir


class TestDataLoader:
    """Tests for DataLoader class."""

    def test_validate_files_exist_all_present(self, temp_data_dir):
        """Test file validation when all files present."""
        loader = DataLoader(temp_data_dir)
        missing = loader.validate_files_exist()
        assert missing == []

    def test_validate_files_exist_missing(self):
        """Test file validation when files missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            missing = loader.validate_files_exist()
            assert len(missing) == 4  # All four files missing

    def test_load_train_success(self, temp_data_dir):
        """Test successful train data loading."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_train(validate=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert TARGET_COLUMN in df.columns
        assert ID_COLUMN in df.columns
        for feat in BASELINE_FEATURES:
            assert feat in df.columns

    def test_load_test_success(self, temp_data_dir):
        """Test successful test data loading."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_test(validate=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert TARGET_COLUMN not in df.columns
        assert ID_COLUMN in df.columns
        for feat in BASELINE_FEATURES:
            assert feat in df.columns

    def test_load_train_missing_file(self):
        """Test loading train when file missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            with pytest.raises(FileNotFoundError):
                loader.load_train()

    def test_validate_train_duplicate_ids(self, temp_data_dir):
        """Test validation catches duplicate IDs in train."""
        # Create train with duplicate IDs
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "Id": [1, 2, 2, 4, 5],  # Duplicate ID
                "GrLivArea": [1500, 2000, 1800, 2200, 1600],
                "BedroomAbvGr": [3, 4, 3, 4, 3],
                "FullBath": [2, 3, 2, 3, 2],
                "SalePrice": [200000, 300000, 250000, 350000, 220000]
            }
            train_path = Path(tmpdir) / "train.csv"
            pd.DataFrame(data).to_csv(train_path, index=False)

            # Create other required files
            for fname in ["test.csv", "data_description.txt", "sample_submission.csv"]:
                Path(tmpdir, fname).touch()

            loader = DataLoader(tmpdir)
            with pytest.raises(ValueError, match="duplicate Id"):
                loader.load_train(validate=True)

    def test_validate_train_missing_target(self, temp_data_dir):
        """Test validation catches missing target values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = {
                "Id": [1, 2, 3, 4, 5],
                "GrLivArea": [1500, 2000, 1800, 2200, 1600],
                "BedroomAbvGr": [3, 4, 3, 4, 3],
                "FullBath": [2, 3, 2, 3, 2],
                "SalePrice": [200000, np.nan, 250000, 350000, 220000]  # NaN in target
            }
            train_path = Path(tmpdir) / "train.csv"
            pd.DataFrame(data).to_csv(train_path, index=False)

            for fname in ["test.csv", "data_description.txt", "sample_submission.csv"]:
                Path(tmpdir, fname).touch()

            loader = DataLoader(tmpdir)
            with pytest.raises(ValueError, match="missing values"):
                loader.load_train(validate=True)

    def test_validate_test_has_target(self, temp_data_dir):
        """Test validation fails if test.csv has target column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Normal train
            train_data = {
                "Id": [1, 2, 3], "GrLivArea": [1500, 2000, 1800],
                "BedroomAbvGr": [3, 4, 3], "FullBath": [2, 3, 2],
                "SalePrice": [200000, 300000, 250000]
            }
            Path(tmpdir, "train.csv").write_text(pd.DataFrame(train_data).to_csv(index=False))

            # Test with target (should fail)
            test_data = {
                "Id": [4, 5], "GrLivArea": [1700, 2100],
                "BedroomAbvGr": [3, 4], "FullBath": [2, 3],
                "SalePrice": [220000, 320000]  # Should not be in test
            }
            Path(tmpdir, "test.csv").write_text(pd.DataFrame(test_data).to_csv(index=False))

            for fname in ["data_description.txt", "sample_submission.csv"]:
                Path(tmpdir, fname).touch()

            loader = DataLoader(tmpdir)
            with pytest.raises(ValueError, match="should not contain target"):
                loader.load_test(validate=True)

    def test_get_baseline_features(self, temp_data_dir):
        """Test extracting baseline features."""
        loader = DataLoader(temp_data_dir)
        train_df = loader.load_train()
        features = loader.get_baseline_features(train_df)

        assert list(features.columns) == BASELINE_FEATURES
        assert len(features) == 5

    def test_get_target(self, temp_data_dir):
        """Test extracting target column."""
        loader = DataLoader(temp_data_dir)
        train_df = loader.load_train()
        target = loader.get_target(train_df)

        assert target.name == TARGET_COLUMN
        assert len(target) == 5

    def test_get_target_missing(self, temp_data_dir):
        """Test get_target fails when target missing."""
        loader = DataLoader(temp_data_dir)
        test_df = loader.load_test()

        with pytest.raises(ValueError, match="not found"):
            loader.get_target(test_df)

    def test_get_ids(self, temp_data_dir):
        """Test extracting ID column."""
        loader = DataLoader(temp_data_dir)
        train_df = loader.load_train()
        ids = loader.get_ids(train_df)

        assert ids.name == ID_COLUMN
        assert list(ids) == [1, 2, 3, 4, 5]

    def test_load_data_description(self, temp_data_dir):
        """Test loading data description file."""
        loader = DataLoader(temp_data_dir)
        desc = loader.load_data_description()

        assert isinstance(desc, str)
        assert "GrLivArea" in desc


class TestConvenienceFunction:
    """Tests for load_and_validate convenience function."""

    def test_load_and_validate(self, temp_data_dir):
        """Test convenience function loads both train and test."""
        from data_loader import load_and_validate

        train_df, test_df = load_and_validate(temp_data_dir)

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) == 5
        assert len(test_df) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])