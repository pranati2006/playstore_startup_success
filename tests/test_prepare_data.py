import pandas as pd

from src.prepare_data import make_success_target


def test_success_target_created():
    df = pd.DataFrame(
        {
            "appId": [f"app.{i}" for i in range(40)],
            "genre": ["Tools"] * 40,
            "realInstalls": [100 * (i + 1) for i in range(40)],
            "reviews": [10 * (i + 1) for i in range(40)],
        }
    )

    result = make_success_target(df)

    assert "success_score" in result.columns
    assert "success" in result.columns
    assert set(result["success"].unique()).issubset({0, 1})
    assert result["success"].sum() > 0
