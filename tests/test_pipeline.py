import os
import sys
import pytest
import pandas as pd
import numpy as np

# Make src/ importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestIngest:
    def test_returns_dataframe(self, tmp_path):
        from ingest import fetch_stock_data
        df = fetch_stock_data("RELIANCE.NS", "2023-01-01", "2023-03-31",
                              output_path=str(tmp_path / "raw.csv"))
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "Close" in df.columns
        assert "Date"  in df.columns

    def test_bad_ticker_returns_empty(self):
        from ingest import fetch_stock_data
        df = fetch_stock_data("NOTAREALTICKERXYZ", "2023-01-01", "2023-06-01")
        assert df.empty

    def test_no_multiindex_columns(self):
        from ingest import fetch_stock_data
        df = fetch_stock_data("TCS.NS", "2023-01-01", "2023-03-31")
        if not df.empty:
            assert not isinstance(df.columns, pd.MultiIndex), \
                "MultiIndex columns were not flattened"


class TestPreprocess:
    def _make_raw_csv(self, tmp_path, n=200):
        np.random.seed(42)
        close  = 2800 + np.cumsum(np.random.randn(n) * 10)
        dates  = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "Date":   dates,
            "Open":   close + np.random.randn(n),
            "High":   close + np.abs(np.random.randn(n)) * 5,
            "Low":    close - np.abs(np.random.randn(n)) * 5,
            "Close":  close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        })
        p = tmp_path / "raw.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_output_has_expected_columns(self, tmp_path):
        from preprocess import preprocess_data, FEATURE_COLS
        raw  = self._make_raw_csv(tmp_path)
        out  = str(tmp_path / "proc.csv")
        preprocess_data(raw, out)
        result = pd.read_csv(out)
        for col in FEATURE_COLS:
            assert col in result.columns, f"Missing column: {col}"

    def test_values_scaled_between_0_and_1(self, tmp_path):
        from preprocess import preprocess_data, FEATURE_COLS
        raw  = self._make_raw_csv(tmp_path)
        out  = str(tmp_path / "proc.csv")
        preprocess_data(raw, out)
        result = pd.read_csv(out)
        for col in FEATURE_COLS:
            assert result[col].min() >= -1e-6, f"{col} min below 0"
            assert result[col].max() <= 1 + 1e-6, f"{col} max above 1"

    def test_no_date_column_in_output(self, tmp_path):
        from preprocess import preprocess_data
        raw = self._make_raw_csv(tmp_path)
        out = str(tmp_path / "proc.csv")
        preprocess_data(raw, out)
        result = pd.read_csv(out)
        assert "Date" not in result.columns

    def test_scaler_saved(self, tmp_path):
        from preprocess import preprocess_data
        import joblib
        raw    = self._make_raw_csv(tmp_path)
        out    = str(tmp_path / "proc.csv")
        scaler = str(tmp_path / "scaler.pkl")
        preprocess_data(raw, out, scaler)
        assert os.path.exists(scaler)
        sc = joblib.load(scaler)
        assert hasattr(sc, "transform")


class TestTrain:
    def _make_processed_csv(self, tmp_path):
        from preprocess import preprocess_data
        np.random.seed(0)
        n     = 250
        close = 2800 + np.cumsum(np.random.randn(n) * 10)
        dates = pd.date_range("2022-01-01", periods=n, freq="B")
        df    = pd.DataFrame({
            "Date":   dates,
            "Open":   close + np.random.randn(n),
            "High":   close + np.abs(np.random.randn(n)) * 5,
            "Low":    close - np.abs(np.random.randn(n)) * 5,
            "Close":  close,
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        })
        raw  = str(tmp_path / "raw.csv")
        proc = str(tmp_path / "proc.csv")
        df.to_csv(raw, index=False)
        preprocess_data(raw, proc)
        return proc

    def test_model_file_created(self, tmp_path):
        from train import train_model
        proc  = self._make_processed_csv(tmp_path)
        model = str(tmp_path / "model.pkl")
        train_model(proc, model)
        assert os.path.exists(model)

    def test_rmse_is_finite(self, tmp_path):
        from train import train_model
        proc  = self._make_processed_csv(tmp_path)
        model = str(tmp_path / "model.pkl")
        rmse  = train_model(proc, model)
        assert np.isfinite(rmse)
        assert rmse >= 0
