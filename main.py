#%%
"""
Author : Abdel YEZZA (Ph.D)
Date :  july 2021
This code is completely free and can be modified without any restriction
Examples given below are presented and explained in my article: 
QUELQUES NOTIONS DE BASE DE LA FINANCE DES PROJETS available at this web address: 
https://www.anigraphics.fr/introduction/math_finance/basic_finance_elements-v3.pdf
"""

import sys
# import Jinja2
# from pandas.io.formats import style
from scipy import optimize
import sympy
# numpy is no more supported and replaced by numpy_financial for some functions
# (see https://pypi.org/project/numpy-financial/), so use numpy_financial instead
import numpy as np
import numpy_financial as npf    
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Create output directories if they don't exist
os.makedirs("output/graphs", exist_ok=True)
os.makedirs("output/data", exist_ok=True)


class FinancialCalculator:
    @staticmethod
    def compute_npv(rate, cash_flows=None):
        """
        Compute Net Present Value
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
        Compute Internal Rate of Return
        :param cash_flows: cash flows data list
        :return: irr value of cash_flows data list
        """
        if cash_flows is None:
            cash_flows = []
        return npf.irr(cash_flows)

    @staticmethod
    def compute_mirr(cash_flows=None, rate=0):
        """
        Compute Modified Internal Rate of Return
        :param cash_flows: cash flows data list
        :param rate: discount rate value
        :return: mirr value of cash_flows data list
        """
        if cash_flows is None:
            cash_flows = []
        return npf.mirr(cash_flows, rate, rate)

    @staticmethod
    def compute_periodic_payment(rate=0.5, n_periods=1, principal=1, fv=0, when='begin'):
        """
        Compute periodic payment for a loan
        Args:
            principal (number): loan principal
            n_periods (number): number of loan duration periods
            rate (float): loan annual rate (between 0 and 1)
            fv (number): future value
            when (str): when payments are made ('begin' or 'end')
        """
        return -npf.pmt(rate, n_periods, principal, fv, when)

    @staticmethod
    def compute_cum_cf(rate, cash_flows=None):
        """
        Compute cumulative cash flow data
        :param rate: discount rate
        :param cash_flows: cash flows list
        :return: cumulative rated cash flow
        """
        if cash_flows is None:
            cash_flows = []
        rated_cf = []
        cum_cf = []
        for cf in cash_flows:
            cum_cf.append(cf)
            rated_cf.append(npf.npv(rate, cum_cf))
        return rated_cf

    @staticmethod
    def compute_pi(rate, cash_flows=None):
        """
        Compute Profitability Index
        :param rate: discount rate
        :param cash_flows: cash flows list
        :return: profitability index value
        """
        if cash_flows is None:
            cash_flows = []
        cf = pd.DataFrame(cash_flows, columns=['CashFlows'])
        cf['DiscountedCashFlows'] = npf.pv(rate=rate, pmt=0, nper=cf.index, fv=-cf['CashFlows'])
        pi_cof_sum = np.cumsum(cf[cf['DiscountedCashFlows'] < 0])
        pi_cif_sum = np.cumsum(cf[cf['DiscountedCashFlows'] >= 0])
        sum_cif = pi_cif_sum.tail(1).DiscountedCashFlows.values[0]
        sum_cof = pi_cof_sum.tail(1).DiscountedCashFlows.values[0]
        if sum_cof != 0:
            return abs(sum_cif)/abs(sum_cof)
        return None


class GraphPlotter:
    @staticmethod
    def plot_2d_graph(x_data=None, y_data=None, plot_params=None):
        """
        Plot 2D graph with customizable parameters
        :param x_data: x-axis data
        :param y_data: y-axis data
        :param plot_params: dictionary of plot parameters
        :return: figure and axes objects
        """
        if x_data is None:
            x_data = []
        if y_data is None:
            y_data = []
        if plot_params is None:
            plot_params = {}

        default_plot_params = {
            'title': '',
            'fontsize': '12',
            'fontname': 'arial',
            'color': '#000000',
            'x_label': 'variable',
            'y_label': 'Value',
            'style': '+-b',
            'x_step': (max(x_data)-min(x_data))/10 if len(x_data) > 1 else 1,
            'filename': 'graph'  # Default filename for saving
        }

        for key in default_plot_params:
            if key not in plot_params or plot_params[key] is None:
                plot_params[key] = default_plot_params[key]

        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False,
                            subplot_kw={'facecolor': 'white'},
                            gridspec_kw={})
        plt.grid(True, which='major', axis='both', lw=1, ls='--', c='.75')
        ax.plot(x_data, y_data, plot_params['style'])
        
        x_ticks = np.arange(x_data[0], x_data[-1] + plot_params['x_step'], plot_params['x_step'])
        ax.set_xticks(x_ticks)
        
        ax.set_xlabel(plot_params['x_label'], labelpad=5, fontsize=14, fontname='serif', color="blue")
        ax.set_ylabel(plot_params['y_label'], labelpad=5, fontsize=14, fontname='serif', color="red")
        
        ax.set_title(plot_params['title'], fontsize=plot_params['fontsize'],
                    fontname=plot_params['fontname'], color=plot_params['color'])

        return fig, ax

    @staticmethod
    def add_annotation_to_graph(fig=None, ax=None, p=None, text='', xytext=None, arrowprops=None):
        """
        Add annotation to graph
        :param fig: matplotlib figure
        :param ax: matplotlib axes
        :param p: point coordinates [x, y]
        :param text: annotation text
        :param xytext: text position
        :param arrowprops: arrow properties
        """
        if ax is None or p is None:
            return
            
        if xytext is None:
            xytext = (p[0]-20, p[1]-20)
            
        if arrowprops is None:
            arrowprops = dict(arrowstyle="->", connectionstyle="arc3, rad=.5")

        if p[1] is not None:
            ax.axhline(p[1], ls='-.')
        if p[0] is not None:
            ax.axvline(p[0], ls='-.')

        ax.annotate(text, fontsize=12, family="serif", xy=p, xycoords="data",
                   textcoords="offset points", xytext=xytext, arrowprops=arrowprops)

    @staticmethod
    def save_figure(fig, filename, dpi=300):
        """
        Save figure to output/graphs directory with timestamp
        :param fig: matplotlib figure to save
        :param filename: base filename without extension
        :param dpi: resolution for the saved figure
        :return: full path to saved file
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with timestamp
        full_filename = f"{filename}_{timestamp}.png"
        
        # Create full path
        full_path = os.path.join("output", "graphs", full_filename)
        
        # Save figure with tight layout to include all elements
        fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
        
        return full_path

    @staticmethod
    def finalize_and_save(fig, filename, show=True):
        """
        Finalize the figure (adjust layout) and save it with all annotations
        :param fig: matplotlib figure
        :param filename: base filename without extension
        :param show: whether to show the figure after saving
        :return: full path to saved file
        """
        # Adjust the layout to make sure all elements are visible
        plt.tight_layout()
        
        # Save the figure with all annotations
        saved_path = GraphPlotter.save_figure(fig, filename)
        
        # Show the figure if requested
        if show:
            fig.show()
            
        return saved_path


class DataManager:
    @staticmethod
    def save_dataframe(df, filename):
        """
        Save pandas DataFrame to output/data directory with timestamp
        :param df: pandas DataFrame to save
        :param filename: base filename without extension
        :return: full path to saved file
        """
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename with timestamp
        full_filename = f"{filename}_{timestamp}.csv"
        
        # Create full path
        full_path = os.path.join("output", "data", full_filename)
        
        # Save DataFrame
        df.to_csv(full_path, index=True)
        
        return full_path


class FinancialExamples:
    @staticmethod
    def example_1_1_1_1():
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
        sol = solve_by_newton_algo(f, 5)
        print('Found solution by Newton algo: x=' + str(sol))
        # alternatively we can use other alogos
        sol = solve_by_bisect_algo(f, .1, 10)
        print('Found solution by BISECT algo: x=' + str(sol))
        sol = solve_by_brentq_algo(f, .1, 10)
        print('Found solution by BRENT algo: x=' + str(sol))

        # add annotation to found solution
        GraphPlotter.add_annotation_to_graph(fig, ax, [sol, 0], 'Equation solution: ' + str(round(sol, 5)),
                                xytext=(sol + 20, 0 + 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        GraphPlotter.finalize_and_save(fig, "example_1_1_1_1_function")

    @staticmethod
    def example_1_1():
        rate = 0.05  # rate = 5%
        cash_flows = [-100000, -20000, 10000, 20000, 50000, 60000, 70000, 80000, 80000, 90000]
        print('\n\nexample_1_1 =================================================================================================')
        print('Exemple 1_1: Calcul de la NPV')
        print('Discount rate: ' + str(rate))
        print('cash flows data: ' + str(cash_flows))
        print('NPV = ' + str(round(FinancialCalculator.compute_npv(rate, cash_flows), 2)))

    @staticmethod
    def example_1_1_1():
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
        pbp = payback_period(cash_flows)
        print('PBP = ' + str(pbp))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Get discounted Payback period (DPBP) value')
        disc_pbp = discounted_payback_period(rate, cash_flows)
        print('DPBP = ' + str(disc_pbp))

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Drawing the cumulative discounted cash flows data graph...')
        fig, ax = GraphPlotter.plot_2d_graph([i for i in range(0, 10, 1)], cum_disc_cash_flows,
                                {'title': 'Cumulative discounted cash flow', 'fontsize': '12', 'fontname': 'arial',
                                        'color': '#000000', 'x_label': 'Periods', 'y_label': 'Discounted cash flow',
                                        'style': 'o-r', 'x_step': None})
        GraphPlotter.add_annotation_to_graph(fig, ax, [disc_pbp, 0], "DPBP point: " + str(round(disc_pbp, 5)),
                                xytext=(disc_pbp - 20, 0 - 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        GraphPlotter.finalize_and_save(fig, "example_1_1_1_cumulative_discounted_cf")

        print('\n\n=================================================================================================')
        print('Exemple 1_1_1: Drawing the cumulative cash flows data graph...')
        fig, ax = GraphPlotter.plot_2d_graph([i for i in range(0, 10, 1)], np.cumsum(cash_flows),
                                {'title': 'Cumulative cash flow', 'fontsize': '12', 'fontname': 'arial',
                                        'color': '#000000', 'x_label': 'Periods', 'y_label': 'Cash flow',
                                        'style': 'o-r', 'x_step': None})
        GraphPlotter.add_annotation_to_graph(fig, ax, [pbp, 0], "PBP point: " + str(round(pbp, 5)),
                                xytext=(pbp - 20, 0 - 50),
                                arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))
        
        # Save the figure with annotations
        GraphPlotter.finalize_and_save(fig, "example_1_1_1_cumulative_cf")

    @staticmethod
    def example_1_4():
        # compute irr
        irr = FinancialCalculator.compute_irr([-400000, 1020000, -630000])
        print('\n\n example_1_4 : =================================================================================================')
        print('irr=' + str(irr))

    @staticmethod
    def example_1_5():
        pi = FinancialCalculator.compute_pi(0.05, [-100000, -20000, 10000, 20000, 50000, -60000, 70000, -80000, 80000, 90000])
        print('\n\n example_1_5: =================================================================================================')
        print('Profitability Index = ' + str(pi))

    @staticmethod
    def example_5_1():
        cf1 = [-10000, 6500, 3000, 3000, 1000]
        rate1 = 12/100
        cf2 = [-10000, 3500, 3500, 3500, 3500]
        rate2 = 12/100
        npv_1 = FinancialCalculator.compute_npv(rate1, cf1)
        npv_2 = FinancialCalculator.compute_npv(rate2, cf2)
        pbp_1 = payback_period(cf1)
        pbp_2 = payback_period(cf2)
        dpbp_1 = discounted_payback_period(rate1, cf1)
        dpbp_2 = discounted_payback_period(rate2, cf2)
        irr_1 = FinancialCalculator.compute_irr(cf1)
        irr_2 = FinancialCalculator.compute_irr(cf2)
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
        fig, ax = GraphPlotter.plot_2d_graph([r for r in np.arange(0.0, 1.1, 0.08)], [npv1(r) for r in np.arange(0.0, 1.1, 0.08)],
                                {'title': 'NPV1\n' + str(expr), 'fontsize': '12', 'fontname': 'arial',
                                 'color': '#000000', 'x_label': '$r$', 'y_label': '$NPV1(r)$', 'style': '.-b',
                                 'x_step': None})
        try:
            print('Try to apply Newton method to our equation :\n' + str(npv1(r)) + ' = 0')
            sol = solve_by_newton_algo(npv1, 0.2)
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
        fig, ax = GraphPlotter.plot_2d_graph([r for r in np.arange(0.0, 1.1, 0.08)], [npv2(r) for r in np.arange(0.0, 1.1, 0.08)],
                                {'title': 'NPV2\n' + str(expr), 'fontsize': '12', 'fontname': 'arial',
                                 'color': '#000000', 'x_label': '$r$', 'y_label': '$NPV2(r)$', 'style': '.-r',
                                 'x_step': None})
        try:
            print('Try to apply Newton method to our equation :\n' + str(npv2(r)) + ' = 0')
            sol = solve_by_newton_algo(npv2, 0.2)
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

    @staticmethod
    def example_5_2():
        cf1 = [-10000, 6500, 3000, 3000, 1000]
        rate1 = 12/100
        cf2 = [-10000, 3500, 3500, 3500, 3500]
        rate2 = 12/100
        npv_1 = FinancialCalculator.compute_npv(rate1, cf1)
        npv_2 = FinancialCalculator.compute_npv(rate2, cf2)
        pbp_1 = payback_period(cf1)
        pbp_2 = payback_period(cf2)
        dpbp_1 = discounted_payback_period(rate1, cf1)
        dpbp_2 = discounted_payback_period(rate2, cf2)
        irr_1 = FinancialCalculator.compute_irr(cf1)
        irr_2 = FinancialCalculator.compute_irr(cf2)
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

    @staticmethod
    def example_7_1():
        print('\n\n example_7_1 =================================================================================================')
        print('Compute periodic payment at begin of period')
        print('periodic payment = ' + str(compute_periodic_payment_at_begin( 6/100 / 12, 5, 1000)))
        print('Compute periodic payment at end of period')
        print('periodic payment = ' + str(compute_periodic_payment_at_end(6/100/12, 5, 1000)))

    @staticmethod
    def example_7_2():
        print('\n\n example_7_2 =================================================================================================')
        print('Compute amortization table data')
        data = get_amortization_data(6/100/12, 48, 100000, 'end')
        df = pd.DataFrame(data, columns=['Period', 'CA', 'Payment', 'Interest', 'RC', 'CR'])
        print(df)

        # Save the DataFrame
        saved_csv = DataManager.save_dataframe(df, "amortization_table_begin")
        print(f"\nAmortization table saved to: {saved_csv}")

        print('\n\nCompute amortization table data')
        data = get_amortization_data(6/100/12, 48, 100000, 'end')
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

    @staticmethod
    def example_7_4():
        print('\n\n example_7_4 =================================================================================================')
        print('Compute amortization table data')
        data = get_amortization_data(0.1, 5, 1000, 'begin')
        df = pd.DataFrame(data, columns=['Period', 'CA', 'Payment', 'Interest', 'RC', 'CR'])
        print(df)


def compute_npv(rate, cash_flows=list()):
    """
    :param rate: discount rate (between 0 and 1)
    :param cash_flows: cash flows data list
    :return: NPV value
    """
    return FinancialCalculator.compute_npv(rate, cash_flows)


def compute_irr(cash_flows=list()):
    """
    :param cash_flows: cash flows data list
    :return: irr value of cash_flows data list
    """
    return FinancialCalculator.compute_irr(cash_flows)


def compute_mirr(cash_flows=list(), rate=0):
    """
    :param cash_flows: cash flows data list
    :param rate: discount rate value
    :return: mirr value of cash_flows data list
    """
    return FinancialCalculator.compute_mirr(cash_flows, rate)


def compute_periodic_payment(rate=0.5, n_periods=1, principal=1, fv=0, when='begin'):
    """This function computes periodic payment for a loan with

    Args:
        principal ([number]): [loan principal]
        n_periods ([number (integer)>0]): [number of loan duration periods]
        rate (int, optional): [loan annual rate (between 0 and 1) ]. Defaults to 0.5 (50%).
    """
    return FinancialCalculator.compute_periodic_payment(rate, n_periods, principal, fv, when)


def compute_periodic_payment_at_begin(rate=0.5, n_periods=2, principal=1, fv=0):
    return compute_periodic_payment(rate, n_periods, principal, fv, 'begin')


def compute_periodic_payment_at_end(rate=0.5, n_periods=2, principal=1, fv=0):
    return compute_periodic_payment(rate, n_periods, principal, fv, 'end')


def get_amortization_data(rate=.5,  n_periods=2, principal=1,when='begin'):
    # get periodic payment
    per_pym = compute_periodic_payment(rate, n_periods, principal, 0, when)
    # compute amortization table data
    if n_periods>0:
        i0 = 0 if when=='begin' else 1
        n_per = n_periods-1 if when=='begin' else n_periods
        CA = principal
        RC = 0
        data = []
        for i in range(i0, n_per+1, 1):
            CA = CA-RC
            I = rate*CA
            RC = per_pym - I
            data.append([i, CA, per_pym, I, RC, CA-RC])
    
    return data
            

def payback_period(cash_flows=list()):
    # convert list to pandas DataFrame object
    cf = pd.DataFrame(cash_flows, columns=['CashFlows'])
    cf.index.name = 'Period'

    # add cum cash flow column as a cum sum
    cf['CumulativeCashFlows'] = np.cumsum(cf['CashFlows'])
    last_period = cf[cf.CumulativeCashFlows < 0].index.values.max()
    ratio = abs(cf.CumulativeCashFlows[last_period]) / \
            (abs(cf.CumulativeCashFlows[last_period]) + abs(cf.CumulativeCashFlows[last_period + 1]))
    pbp = last_period + ratio
    return pbp


def discounted_payback_period(rate, cash_flows=list()):
    # convert list to pandas DataFrame object
    cf = pd.DataFrame(cash_flows, columns=['CashFlows'])
    cf.index.name = 'Period'
    # compute DiscountedCashFlows column values
    cf['DiscountedCashFlows'] = npf.pv(rate=rate, pmt=0, nper=cf.index, fv=-cf['CashFlows'])
    # add cum discounted cash flow column as a cum sum
    cf['CumulativeDiscountedCashFlows'] = np.cumsum(cf['DiscountedCashFlows'])
    last_period = cf[cf.CumulativeDiscountedCashFlows < 0].index.values.max()
    ratio = abs(cf.CumulativeDiscountedCashFlows[last_period])/\
            (abs(cf.CumulativeDiscountedCashFlows[last_period])+cf.CumulativeDiscountedCashFlows[last_period + 1])
    discounted_pbp = last_period + ratio
    return discounted_pbp


def compute_cum_cf(rate, cash_flows=list()):
    """
    Compute cumulative cash flow data from cash flow
    :param rate
    :param cash_flows:
    :return: cumulative rated cash flow
    """
    return FinancialCalculator.compute_cum_cf(rate, cash_flows)


def plot_2d_graph(x_data=list(), y_data=list(), plot_params = None):
    return GraphPlotter.plot_2d_graph(x_data, y_data, plot_params)


def add_annotation_to_graph(fig=None, ax=None, p=None, text='', xytext=None, arrowprops=None):
    """
    add text and arrows as annotation to graph
    fig: figure
    ax: axis
    p: point of arrow end
    xytext: start position of text
    text: text
    arrowprops: arrow properties as dictionary
    vertical and horizontal lines are added with intersection point p
    """
    return GraphPlotter.add_annotation_to_graph(fig, ax, p, text, xytext, arrowprops)


def compute_pi(rate, cash_flows=list()):
    return FinancialCalculator.compute_pi(rate, cash_flows)


def solve_by_newton_algo(f, x0):
    # solve equation f(x)=0 with respect to x variable and x0 as initial guess value
    return optimize.newton(f, x0)


def solve_by_bisect_algo(f, a, b, x_tol=0.000001):
    # solve equation f(x)=0 using bisection method with respect to x variable limited 
    # to interval [a, b] with a tolerance of x_tol
    return optimize.bisect(f, a, b, xtol=x_tol)


def solve_by_brentq_algo(f, a, b, x_tol=0.000001):
    # solve equation f(x)=0 using "Brent's method with respect to x variable 
    # limited to interval [a, b] with a tolerance of x_tol
    return optimize.brentq(f, a, b, xtol=x_tol)        


def my_main():
    # run all examples
    examples = FinancialExamples()
    examples.example_1_1_1_1()
    examples.example_1_1()
    examples.example_1_1_1()
    examples.example_1_4()
    examples.example_1_5()
    examples.example_5_1()
    examples.example_5_2()
    examples.example_7_1()
    examples.example_7_2()
    examples.example_7_4()
    return 0


if __name__ == '__main__':
    sys.exit(my_main())