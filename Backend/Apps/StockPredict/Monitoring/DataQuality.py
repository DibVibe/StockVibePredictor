import logging
from typing import Dict, List
import pandas as pd

logger = logging.getLogger(__name__)

class DataQualityMonitor:
    """Monitor and report data quality issues"""

    @staticmethod
    def check_data_quality(data: pd.DataFrame, ticker: str) -> Dict:
        """Comprehensive data quality check"""
        issues = []
        metrics = {}

        # Check for missing data
        missing_pct = (data.isnull().sum() / len(data) * 100).to_dict()
        if any(pct > 5 for pct in missing_pct.values()):
            issues.append(f"High missing data: {missing_pct}")

        # Check for data gaps
        if not data.empty:
            time_diffs = data.index.to_series().diff()
            max_gap = time_diffs.max()
            if pd.notna(max_gap) and max_gap.total_seconds() > 86400:
                issues.append(f"Data gap detected: {max_gap}")

        # Check for outliers
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data.columns:
                z_scores = abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > 3
                if outliers.any():
                    issues.append(f"Outliers in {col}: {outliers.sum()} points")

        metrics['total_rows'] = len(data)
        metrics['date_range'] = f"{data.index[0]} to {data.index[-1]}" if not data.empty else "N/A"
        metrics['quality_score'] = 100 - len(issues) * 10

        if issues:
            logger.warning(f"Data quality issues for {ticker}: {issues}")

        return {
            'issues': issues,
            'metrics': metrics,
            'is_healthy': len(issues) == 0
        }
