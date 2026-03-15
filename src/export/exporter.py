"""
Export module — CSV and Excel export for dashboard data.
"""

import io
import pandas as pd
from datetime import datetime


def to_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode("utf-8")


def to_excel(dataframes: dict) -> bytes:
    """
    Convert multiple DataFrames to Excel with one sheet per query.

    Args:
        dataframes: dict of {sheet_name: DataFrame}

    Returns:
        Excel file as bytes
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, df in dataframes.items():
            safe_name = sheet_name[:31]  # Excel sheet name limit
            # Strip timezone from datetime columns (Excel requirement)
            df = df.copy()
            for col in df.select_dtypes(include=["datetimetz"]).columns:
                df[col] = df[col].dt.tz_localize(None)
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return output.getvalue()


def export_filename(prefix: str, extension: str) -> str:
    """Generate timestamped export filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}.{extension}"