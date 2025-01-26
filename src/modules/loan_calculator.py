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

        print(f"Loan parameters (rate, n_periods, principal, when)=({self.rate}, {self.n_periods}, {self.principal}, {self.when})")

    
    def compute_periodic_payment(self):
        monthly_rate = self.rate
        total_payments = self.n_periods
        
        if self.when == 'end':
            monthly_payment = (self.principal * monthly_rate * (1 + monthly_rate) ** total_payments) / ((1 + monthly_rate) ** total_payments - 1)
        elif self.when == 'begin':
            payment_end = (self.principal * monthly_rate * (1 + monthly_rate) ** total_payments) / ((1 + monthly_rate) ** total_payments - 1)
            monthly_payment = payment_end / (1 + monthly_rate)

        print(f"Monthly payment: {monthly_payment}")
        
        return monthly_payment


    
    def get_amortization_data(self):
        """Get amortization schedule data
        Returns:
            list: List of lists containing [period, current_amount, periodic_payment, interest, repayment_capital, remaining_amount]
        """
        R = self.compute_periodic_payment()
        schedule = []
        
        CA = self.principal
        print(f"Initial capital: {CA}")
        print(f"Payment timing: {self.when}") if self.when else "unknown"
        RC = 0 
        for i in range(1, self.n_periods + 1):
            CA = CA - RC
            if CA < 0:
                CA = 0
            
            if self.when == 'begin' and i == 1:
                #print("when == 'begin' and i == 1")
                RC = R
                interest_payment = 0
            else:
                #print("when != 'begin' or i != 1")
                interest_payment = CA * self.rate
                RC = R - interest_payment
            
            #print("RC = ", RC)
            #print("interest_payment = ", interest_payment)

            schedule.append((i, CA , R, interest_payment, R-interest_payment, CA-(R-interest_payment)))
        
        return schedule
    

    def print_amortization_schedule(self):
        print(f"{'Periode':<10}{'Current amount':<20}{'period_payment':<20}{'Interest':<20}{'Non amortized payment':<20}{'Remaining amount':<20}")
        for row in self.get_amortization_data():
            print(f"{row[0]:<10}{row[1]:<20.2f}{row[2]:<20.2f}{row[3]:<20.2f}{row[4]:<20.2f}{row[5]:<20.2f}")
        
        print(f"\n\nAmortization schedule for a payment in {self.when} of periods:")
        self.print_amortization_schedule()
