"""
Loan calculator module providing loan-related calculations.
"""

import numpy_financial as npf
import pandas as pd


class LoanCalculator:
    def __init__(self, rate=0.5, n_periods=1, principal=1, when='begin'):
        """Initialize LoanCalculator with loan parameters
        Args:
            rate (float): loan annual rate (between 0 and 1). Defaults to 0.5 (50%).
            n_periods (int): number of loan duration periods. Defaults to 1.
            principal (float): loan principal. Defaults to 1.
            when (str): when payments are made ('begin' or 'end'). Defaults to 'begin'.
        """
        self.rate = rate
        self.n_periods = n_periods
        self.principal = principal
        self.when = when

    def compute_periodic_payment(self, fv=0):
        """This function computes periodic payment for a loan
        Args:
            fv (float): future value. Defaults to 0.
        Returns:
            float: periodic payment amount
        """
        return -npf.pmt(self.rate, self.n_periods, self.principal, fv, self.when)

    def compute_periodic_payment_at_begin(self, fv=0):
        """Compute periodic payment with payments at the beginning of each period
        Args:
            fv (float): future value. Defaults to 0.
        Returns:
            float: periodic payment amount
        """
        temp_when = self.when
        self.when = 'begin'
        result = self.compute_periodic_payment(fv)
        self.when = temp_when
        return result

    def compute_periodic_payment_at_end(self, fv=0):
        """Compute periodic payment with payments at the end of each period
        Args:
            fv (float): future value. Defaults to 0.
        Returns:
            float: periodic payment amount
        """
        temp_when = self.when
        self.when = 'end'
        result = self.compute_periodic_payment(fv)
        self.when = temp_when
        return result

    def get_amortization_data(self):
        """Get amortization schedule data
        Returns:
            list: List of lists containing [period, current_amount, periodic_payment, interest, repayment_capital, remaining_amount]
        """
        per_pym = self.compute_periodic_payment()
        if self.n_periods > 0:
            i0 = 0 if self.when == 'begin' else 1
            n_per = self.n_periods-1 if self.when == 'begin' else self.n_periods
            CA = self.principal
            RC = 0
            data = []
            for i in range(i0, n_per+1, 1):
                CA = CA-RC
                I = self.rate*CA
                RC = per_pym - I
                data.append([i, CA, per_pym, I, RC, CA-RC])
            return data
        return []
