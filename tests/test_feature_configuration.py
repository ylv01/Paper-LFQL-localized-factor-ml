from __future__ import annotations

import unittest

from src.common import lfql_common
from src.experiments import localized_factor_ml


EXPECTED_FEATURES = [
    "bp",
    "ep",
    "sp",
    "cfp",
    "reversal_5d",
    "momentum_60d",
    "volatility_20d",
    "volatility_60d",
    "skewness_60d",
    "turnover_rate",
    "avg_money_20d",
    "money_volatility_20d",
    "avg_volume_20d",
    "log_total_mv",
    "log_circ_mv",
    "roa",
    "gross_profit_margin",
    "net_profit_margin",
    "revenue_yoy",
    "operating_cash_flow_to_total_assets",
    "debt_to_assets",
]


class FeatureConfigurationTest(unittest.TestCase):
    def test_numeric_feature_contract(self) -> None:
        self.assertEqual(lfql_common.NUMERIC_FEATURES, EXPECTED_FEATURES)
        self.assertEqual(localized_factor_ml.NUMERIC_FEATURES, EXPECTED_FEATURES)
        self.assertEqual(localized_factor_ml.CONDITION_FACTORS, EXPECTED_FEATURES)
        self.assertEqual(len(EXPECTED_FEATURES), 21)

    def test_model_feature_count_contract(self) -> None:
        industry_dummy_count = 32
        self.assertEqual(len(EXPECTED_FEATURES) + industry_dummy_count, 53)


if __name__ == "__main__":
    unittest.main()
