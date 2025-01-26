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
import argparse
from modules.financial_examples import FinancialExamples


def main(args):
    # get the command line arguments
    examples = FinancialExamples() if args is None  else FinancialExamples(rate=float(args.rate)/100/12, n_periods=int(args.n_periods), 
                                     principal=float(args.principal), when=args.when)
    if args is not None:
        print(args)
    
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
    
    if (len(sys.argv) <= 2):
        # call the main function
        if len(sys.argv) == 2 :
            if sys.argv[1] == '--examples':
                sys.exit(main(args=None))
            else:
                print('Unknown command : ', sys.argv[1])
                print('Available commands : \n\teg. python main.py --examples')
                sys.exit(1)
        else:
            sys.exit(main(args=None))
    elif (len(sys.argv) > 1):
        # get the command line argument
        parser = argparse.ArgumentParser(description='Compute amortization table data')
        parser.add_argument('--rate', default=6, type=float, help='Annual interest rate as % value (ex: 6 for 6%)')
        parser.add_argument('--n_periods', default=48, type=int, help='Number of loan periods')
        parser.add_argument('--principal', default=100000, type=float, help='Loan principal')
        parser.add_argument('--when', default='end', choices=['begin', 'end'], 
                            help='When payments are made')
        
        # call the main function
        args = parser.parse_args()
        print(args)
        sys.exit(main(args))

