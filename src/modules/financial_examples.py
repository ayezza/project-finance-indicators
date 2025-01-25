"""
Financial examples module containing example calculations and demonstrations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy

from .financial_calculator import FinancialCalculator
from .loan_calculator import LoanCalculator
from .graph_plotter import GraphPlotter
from .data_manager import DataManager


class FinancialExamples:
    def __init__(self):
        self.fin_calc = FinancialCalculator()
        self.loan_calc = LoanCalculator(rate=6/100/12, n_periods=48, principal=100000, when='end')
        self.plotter = GraphPlotter()

    def example_1_1_1_1(self):
        print('\n\nexample_1_1_1_1 =================================================================================================')
        # define our function
        def f(x):
            return -10 * x ** 9 - 2 * x ** 8 + x ** 7 + 2 * x ** 6 + 5 * x ** 5 + 6 * x ** 4 + \
                   7 * x ** 3 + 8 * x ** 2 + 8 * x + 9

        # plot corresponding graph
        x = sympy.Symbol("x")
        expr = r'$f(x) = -10x^{9} - 2x^{8} + x^{7} + 2x^{6} + 5x ^{5} + 6x^{4} + 7x^{3} + 8x^{2} + 8x + 9$'
        print(expr)
        fig, ax = GraphPlotter.plot_2d_graph([x for x in np.arange(1.2, 1.3, .01)], [f(x) for x in np.arange(1.2, 1.3, .01)],
                                {'title': 'Function graph\n' + str(expr), 'fontsize': '12', 'fontname': 'arial',
                                        'color': '#000000', 'x_label': '$x$', 'y_label': '$f(x)$', 'style': '+-b', 'x_step': None})
        print('Apply Newton method to our equation :\n' + str(f(x)) + ' = 0')
        sol = self.fin_calc.solve_by_newton_algo(f, 5)
        print('Found solution by Newton algo: x=' + str(sol))
        # alternatively we can use other alogos
        sol = self.fin_calc.solve_by_bisect_algo(f, .1, 10)
        print('Found solution by BISECT algo: x=' + str(sol))
        sol = self.fin_calc.solve_by_brentq_algo(f, .1, 10)
        print('Found solution by BRENT algo: x=' + str(sol))

        # add annotation to found solution
        GraphPlotter.add_annotation_to_graph(fig, ax, [sol, 0], 'Equation solution: ' + str(round(sol, 5)),
                                xytext=(sol + 20, 0 + 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        GraphPlotter.finalize_and_save(fig, "example_1_1_1_1_function")

    def example_1_1(self):
        rate = 0.05  # rate = 5%
        cash_flows = [-100000, -20000, 10000, 20000, 50000, 60000, 70000, 80000, 80000, 90000]
        print('\n\nexample_1_1 =================================================================================================')
        print('Exemple 1_1: Calcul de la NPV')
        print('Discount rate: ' + str(rate))
        print('cash flows data: ' + str(cash_flows))
        print('NPV = ' + str(round(FinancialCalculator.compute_npv(rate, cash_flows), 2)))

    def example_1_1_1(self):
        cash_flows = [-100000, -20000, 10000, 20000, 50000, 60000, 70000, 80000, 80000, 90000]
        rate = 0.05  # 5% rate
        cum_disc_cash_flows = FinancialCalculator.compute_cum_cf(rate, cash_flows)
        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Calcul du IRR')
        print('cash flows data: ' + str(cash_flows))
        print('IRR = ' + str(round(FinancialCalculator.compute_irr(cash_flows), 5)))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Calcul du MIRR')
        print('cash flows data: ' + str(cash_flows))
        print('rate value: ' + str(rate))
        print('MIRR = ' + str(round(FinancialCalculator.compute_mirr(cash_flows, rate), 5)))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Get Payback period (PBP) value')
        pbp = self.fin_calc.payback_period(cash_flows)
        print('PBP = ' + str(pbp))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Get discounted Payback period (DPBP) value')
        disc_pbp = self.fin_calc.discounted_payback_period(rate, cash_flows)
        print('DPBP = ' + str(disc_pbp))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Drawing the cumulative discounted cash flows data graph...')
        fig, ax = self.plotter.plot_2d_graph([i for i in range(0, 10, 1)], cum_disc_cash_flows,
                                {'title': 'Cumulative discounted cash flow', 'fontsize': '12', 'fontname': 'arial',
                                        'color': '#000000', 'x_label': 'Periods', 'y_label': 'Discounted cash flow',
                                        'style': 'o-r', 'x_step': None})
        GraphPlotter.add_annotation_to_graph(fig, ax, [disc_pbp, 0], "DPBP point: " + str(round(disc_pbp, 5)),
                                xytext=(disc_pbp - 20, 0 - 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        self.plotter.finalize_and_save(fig, "example_1_1_1_cumulative_discounted_cf")

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Drawing the cumulative cash flows data graph...')
        fig, ax = self.plotter.plot_2d_graph([i for i in range(0, 10, 1)], np.cumsum(cash_flows),
                                {'title': 'Cumulative cash flow', 'fontsize': '12', 'fontname': 'arial',
                                        'color': '#000000', 'x_label': 'Periods', 'y_label': 'Cash flow',
                                        'style': 'o-r', 'x_step': None})
        self.plotter.add_annotation_to_graph(fig, ax, [pbp, 0], "PBP point: " + str(round(pbp, 5)),
                                xytext=(pbp - 20, 0 - 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        self.plotter.finalize_and_save(fig, "example_1_1_1_cumulative_cf")

    
    def example_1_4(self):
        # compute irr
        irr = self.fin_calc.compute_irr([-400000, 1020000, -630000])
        print('\n\n example_1_4 : =================================================================================================')
        print('irr=' + str(irr))

    
    def example_1_5(self):
        pi = self.fin_calc.compute_pi(0.05, [-100000, -20000, 10000, 20000, 50000, -60000, 70000, -80000, 80000, 90000])
        print('\n\n example_1_5: =================================================================================================')
        print('Profitability Index = ' + str(pi))

    
    def example_5_1(self):
        cf1 = [-10000, 6500, 3000, 3000, 1000]
        rate1 = 12/100
        cf2 = [-10000, 3500, 3500, 3500, 3500]
        rate2 = 12/100
        npv_1 = self.fin_calc.compute_npv(rate1, cf1)
        npv_2 = self.fin_calc.compute_npv(rate2, cf2)
        pbp_1 = self.fin_calc.payback_period(cf1)
        pbp_2 = self.fin_calc.payback_period(cf2)
        dpbp_1 = self.fin_calc.discounted_payback_period(rate1, cf1)
        dpbp_2 = self.fin_calc.discounted_payback_period(rate2, cf2)
        irr_1 = self.fin_calc.compute_irr(cf1)
        irr_2 = self.fin_calc.compute_irr(cf2)
        mirr_1 = FinancialCalculator.compute_mirr(cf1, rate1)
        mirr_2 = FinancialCalculator.compute_mirr(cf2, rate2)
        pi_1 = FinancialCalculator.compute_pi(rate1, cf1)
        pi_2 = FinancialCalculator.compute_pi(rate2, cf2)
        print('\n\n example_5_1 =================================================================================================')
        print('NPV1: ' + str(npv_1))
        print('NPV2: ' + str(npv_2))
        print('PBP1: ' + str(pbp_1))
        print('PBP2: ' + str(pbp_2))
        print('DPBP1: ' + str(dpbp_1))
        print('DPBP2: ' + str(dpbp_2))
        print('IRR1: ' + str(irr_1))
        print('IRR2: ' + str(irr_2))
        print('MIRR1: ' + str(mirr_1))
        print('MIRR2: ' + str(mirr_2))
        print('PI1: ' + str(pi_1))
        print('PI2: ' + str(pi_2))

        # plot NPV1 and NPV2 as functions of discount rate r
        def npv1(r):
            return -10000 + 6500/(1 + r) + 3000/((1 + r))**2 + 3000/((1 + r))**3 + 1000/((1 + r))**4

        r = sympy.Symbol("r")
        # NPV1 graph
        expr = r'$NPV1(r) = -10000 +\frac{6500}{(1+r)} + \frac{3000}{(1+r)^2} + \frac{3000}{(1+r)^3} + \frac{3000}{(1+r)^4}$'
        fig, ax = self.plotter.plot_2d_graph([r for r in np.arange(0.0, 1.1, 0.08)], [npv1(r) for r in np.arange(0.0, 1.1, 0.08)],
                                {'title': 'NPV1\n' + str(expr), 'fontsize': '12', 'fontname': 'arial',
                                 'color': '#000000', 'x_label': '$r$', 'y_label': '$NPV1(r)$', 'style': '.-b',
                                 'x_step': None})
        try:
            print('Try to apply Newton method to our equation :\n' + str(npv1(r)) + ' = 0')
            sol = self.fin_calc.solve_by_newton_algo(npv1, 0.2)
            print('Found solution: r=' + str(sol))
            # add annotation to found solution
            GraphPlotter.add_annotation_to_graph(fig, ax, [sol, 0], 'Equation solution: ' + str(round(sol, 5)),
                                    xytext=(sol + 20, 0 + 50),
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        except:
            print('No solution found for NPV1!')
        finally:
            # Save the figure with annotations
            GraphPlotter.finalize_and_save(fig, "example_5_1_npv1")

        # NPV2 graph
        def npv2(r):
            return -10000 + 3500 / (1 + r) + 3500 / ((1 + r)) ** 2 + 3500 / ((1 + r)) ** 3 + 3500 / ((1 + r)) ** 4

        expr = r'$NPV2(r) = -10000 +\frac{3500}{(1+r)} + \frac{3500}{(1+r)^2} + \frac{3500}{(1+r)^3} + \frac{3500}{(1+r)^4}$'
        fig, ax = self.plotter.plot_2d_graph([r for r in np.arange(0.0, 1.1, 0.08)], [npv2(r) for r in np.arange(0.0, 1.1, 0.08)],
                                {'title': 'NPV2\n' + str(expr), 'fontsize': '12', 'fontname': 'arial',
                                 'color': '#000000', 'x_label': '$r$', 'y_label': '$NPV2(r)$', 'style': '.-r',
                                 'x_step': None})
        try:
            print('Try to apply Newton method to our equation :\n' + str(npv2(r)) + ' = 0')
            sol = self.fin_calc.solve_by_newton_algo(npv2, 0.2)
            print('Found solution: r=' + str(sol))
            # add annotation to found solution
            GraphPlotter.add_annotation_to_graph(fig, ax, [sol, 0], 'Equation solution: ' + str(round(sol, 5)),
                                    xytext=(sol + 20, 0 + 50),
                                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        except:
            print('No solution found for NPV2!')
        finally:
            # Save the figure with annotations
            GraphPlotter.finalize_and_save(fig, "example_5_1_npv2")

    
    def example_5_2(self):
        cf1 = [-10000, 6500, 3000, 3000, 1000]
        rate1 = 12/100
        cf2 = [-10000, 3500, 3500, 3500, 3500]
        rate2 = 12/100
        npv_1 = self.fin_calc.compute_npv(rate1, cf1)
        npv_2 = self.fin_calc.compute_npv(rate2, cf2)
        pbp_1 = self.fin_calc.payback_period(cf1)
        pbp_2 = self.fin_calc.payback_period(cf2)
        dpbp_1 = self.fin_calc.discounted_payback_period(rate1, cf1)
        dpbp_2 = self.fin_calc.discounted_payback_period(rate2, cf2)
        irr_1 = self.fin_calc.compute_irr(cf1)
        irr_2 = self.fin_calc.compute_irr(cf2)
        mirr_1 = FinancialCalculator.compute_mirr(cf1, rate1)
        mirr_2 = FinancialCalculator.compute_mirr(cf2, rate2)
        pi_1 = FinancialCalculator.compute_pi(rate1, cf1)
        pi_2 = FinancialCalculator.compute_pi(rate2, cf2)
        print('\n\n example_5_2 =================================================================================================')
        print('NPV1: ' + str(npv_1))
        print('NPV2: ' + str(npv_2))
        print('PBP1: ' + str(pbp_1))
        print('PBP2: ' + str(pbp_2))
        print('DPBP1: ' + str(dpbp_1))
        print('DPBP2: ' + str(dpbp_2))
        print('IRR1: ' + str(irr_1))
        print('IRR2: ' + str(irr_2))
        print('MIRR1: ' + str(mirr_1))
        print('MIRR2: ' + str(mirr_2))
        print('PI1: ' + str(pi_1))
        print('PI2: ' + str(pi_2))

    
    def example_7_1(self):
        print('\n\n example_7_1 =================================================================================================')
        print('Compute periodic payment at begin of period')
        print('periodic payment = ' + str(self.loan_calc.compute_periodic_payment_at_begin(fv=0)))   
        print('Compute periodic payment at end of period')
        print('periodic payment = ' + str(self.loan_calc.compute_periodic_payment_at_end(fv=0)))

    def example_7_2(self):
        print('\n\n example_7_2 =================================================================================================')
        print('Compute amortization table data')
        data = self.loan_calc.get_amortization_data()
        df = pd.DataFrame(data, columns=['Period', 'CA', 'Payment', 'Interest', 'RC', 'CR'])
        print(df)

        # Save the DataFrame
        saved_csv = DataManager.save_dataframe(df, "amortization_table_begin")
        print(f"\nAmortization table saved to: {saved_csv}")

        print('\n\nCompute amortization table data')
        data = self.loan_calc.get_amortization_data()
        df = pd.DataFrame(data, columns=['Period', 'CA', 'Payment', 'Interest', 'RC', 'CR'])
        print(df)

        # Save the second DataFrame
        saved_csv = DataManager.save_dataframe(df, "amortization_table_end")
        print(f"\nAmortization table saved to: {saved_csv}")

        print('Drawing amortization data...')
        fig = plt.figure(figsize=(10, 12))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        pd_data = df  # Using the last DataFrame (end period)
        pd_data.plot('Period', ['RC', 'Interest'], kind='bar', xlabel='', xlim=[1, 48],
                    title='Evolution des éléments RC (Remboursement du Capital) et I (Intérêts)', 
                    ax=ax1, style='o-', fontsize=9, stacked=False)
        pd_data.plot('Period', ['CA', 'CR'], kind='bar', xlabel='Mois', xlim=[1, 48],
                    title='Evolution des éléments CA (Capital à Amortir) et CR (Capital Restant)', 
                    ax=ax2, style='x-', fontsize=9, stacked=False)
        
        # Save the figure with annotations
        GraphPlotter.finalize_and_save(fig, "amortization_evolution")

    
    def example_7_4(self):
        print('\n\n example_7_4 =================================================================================================')
        print('Compute amortization table data')
        data = self.loan_calc.get_amortization_data()
        df = pd.DataFrame(data, columns=['Period', 'CA', 'Payment', 'Interest', 'RC', 'CR'])
        print(df)
