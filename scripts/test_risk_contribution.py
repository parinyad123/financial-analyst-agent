"""
Standalone test: Risk Contribution Analysis
Calls _portfolio_risk_logic directly from src/ — no notebook, no agent.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.portfolio_risk import _portfolio_risk_logic

TEST_PORTFOLIO = '{"NVDA": 5000, "AMD": 3000}'

print(f"Portfolio: {TEST_PORTFOLIO}")
print("=" * 60)
result = _portfolio_risk_logic(TEST_PORTFOLIO)
print(result)
