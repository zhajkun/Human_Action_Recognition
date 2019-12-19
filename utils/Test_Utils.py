# !interpreter [optional-arg]
# -*- coding: utf-8 -*-
# Version 

"""
{Test the utils folder}
{License_info}
"""
import sys, os, time, argparse, logging

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

def SimplePrint():
    print('This Module has been successfully imported')
    print()


def fib(n):    # write Fibonacci series up to n
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        a, b = b, a+b
    print()