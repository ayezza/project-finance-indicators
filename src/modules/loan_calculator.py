"""
Loan calculator module providing loan-related calculations.
"""

import numpy_financial as npf
import pandas as pd


class LoanCalculator:
    @staticmethod
    def compute_periodic_payment(rate=0.5, n_periods=1, principal=1, fv=0, when='begin'):
        """This function computes periodic payment for a loan with
        Args:
            principal ([number]): [loan principal]
            n_periods ([number (integer)>0]): [number of loan duration periods]
            rate (int, optional): [loan annual rate (between 0 and 1) ]. Defaults to 0.5 (50%).
        """
        return -npf.pmt(rate, n_periods, principal, fv, when)

    @staticmethod
    def compute_periodic_payment_at_begin(rate=0.5, n_periods=2, principal=1, fv=0):
        return LoanCalculator.compute_periodic_payment(rate, n_periods, principal, fv, 'begin')

    @staticmethod
    def compute_periodic_payment_at_end(rate=0.5, n_periods=2, principal=1, fv=0):
        return LoanCalculator.compute_periodic_payment(rate, n_periods, principal, fv, 'end')

    @staticmethod
    def get_amortization_data(rate=.5, n_periods=2, principal=1, when='begin'):
        per_pym = LoanCalculator.compute_periodic_payment(rate, n_periods, principal, 0, when)
        if n_periods > 0:
            i0 = 0 if when == 'begin' else 1
            n_per = n_periods-1 if when == 'begin' else n_periods
            CA = principal
            RC = 0
            data = []
            for i in range(i0, n_per+1, 1):
                CA = CA-RC
                I = rate*CA
                RC = per_pym - I
                data.append([i, CA, per_pym, I, RC, CA-RC])
            return data
        return []
