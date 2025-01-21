"""
Main program for financial indicators calculations.
Author : Abdel YEZZA (Ph.D)
Date :  july 2021
This code is completely free and can be modified without any restriction
Examples given below are presented and explained in my article: 
QUELQUES NOTIONS DE BASE DE LA FINANCE DES PROJETS available at this web address: 
https://www.anigraphics.fr/introduction/math_finance/basic_finance_elements-v3.pdf
"""

import sys
from modules.financial_examples import FinancialExamples


def main():
    examples = FinancialExamples()
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
    sys.exit(main())
