from application import Application
from vmbpy import *

def print_preamble():
    print('////////////////////////////////////////')
    print('/// VmbPy Multithreading Example ///////')
    print('////////////////////////////////////////\n')
    print(flush=True)

if __name__ == '__main__':
    print_preamble()
    app = Application()
    app.run()
