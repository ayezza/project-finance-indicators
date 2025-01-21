"""
Financial calculator module providing basic financial calculations.
"""

import numpy as np
import numpy_financial as npf
import pandas as pd
from scipy import optimize


class FinancialCalculator:
    @staticmethod
    def compute_npv(rate, cash_flows=None):
        """
        :param rate: discount rate (between 0 and 1)
        :param cash_flows: cash flows data list
        :return: NPV value
        """
        if cash_flows is None:
            cash_flows = []
        if not (0 <= rate <= 1):
            return 0
        return npf.npv(rate, cash_flows)

    @staticmethod
    def compute_irr(cash_flows=None):
        """
        :param cash_flows: cash flows data list
        :return: irr value of cash_flows data list
        """
        if cash_flows is None:
            cash_flows = []
        return npf.irr(cash_flows)

    @staticmethod
    def compute_mirr(cash_flows=None, rate=0):
        """
        :param cash_flows: cash flows data list
        :param rate: discount rate value
        :return: mirr value of cash_flows data list
        """
        if cash_flows is None:
            cash_flows = []
        return npf.mirr(cash_flows, rate, rate)

    @staticmethod
    def compute_cum_cf(rate, cash_flows=None):
        """
        Compute cumulative cash flow data from cash flow
        :param rate
        :param cash_flows:
        :return: cumulative rated cash flow
        """
        if cash_flows is None:
            cash_flows = []
        rated_cf, cum_cf = [], []
        for cf in cash_flows:
            cum_cf.append(cf)
            rated_cf.append(npf.npv(rate, cum_cf))
        return rated_cf

    @staticmethod
    def payback_period(cash_flows=None):
        if cash_flows is None:
            cash_flows = []
        cf = pd.DataFrame(cash_flows, columns=['CashFlows'])
        cf.index.name = 'Period'
        cf['CumulativeCashFlows'] = np.cumsum(cf['CashFlows'])
        last_period = cf[cf.CumulativeCashFlows < 0].index.values.max()
        ratio = abs(cf.CumulativeCashFlows[last_period]) / \
                (abs(cf.CumulativeCashFlows[last_period]) + abs(cf.CumulativeCashFlows[last_period + 1]))
        return last_period + ratio

    @staticmethod
    def discounted_payback_period(rate, cash_flows=None):
        if cash_flows is None:
            cash_flows = []
        cf = pd.DataFrame(cash_flows, columns=['CashFlows'])
        cf.index.name = 'Period'
        cf['DiscountedCashFlows'] = npf.pv(rate=rate, pmt=0, nper=cf.index, fv=-cf['CashFlows'])
        cf['CumulativeDiscountedCashFlows'] = np.cumsum(cf['DiscountedCashFlows'])
        last_period = cf[cf.CumulativeDiscountedCashFlows < 0].index.values.max()
        ratio = abs(cf.CumulativeDiscountedCashFlows[last_period])/\
                (abs(cf.CumulativeDiscountedCashFlows[last_period])+cf.CumulativeDiscountedCashFlows[last_period + 1])
        return last_period + ratio

    @staticmethod
    def solve_by_newton_algo(f, x0):
        return optimize.newton(f, x0)

    @staticmethod
    def solve_by_bisect_algo(f, a, b, x_tol=0.000001):
        return optimize.bisect(f, a, b, xtol=x_tol)

    @staticmethod
    def solve_by_brentq_algo(f, a, b, x_tol=0.000001):
        return optimize.brentq(f, a, b, xtol=x_tol)

    @staticmethod
    def compute_pi(rate, cash_flows=None):
        if cash_flows is None:
            cash_flows = []
        if len(cash_flows) < 2:
            return 0
        
        initial_investment = abs(cash_flows[0])
        if initial_investment == 0:
            return 0
            
        future_cash_flows = cash_flows[1:]
        npv_future = npf.npv(rate, future_cash_flows)
        return npv_future / initial_investment
