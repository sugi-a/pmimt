import sys

import numpy as np

def main():
    x = 0
    x2 = 0
    m = 10**10
    M = 0
    n = 0
    n_empty = 0

    for line in sys.stdin:
        l = len(line.split())
        if l == 0:
            n_empty += 1
            continue

        x += l
        x2 += l ** 2
        m = min(m, l)
        M = max(M, l)
        n += 1

    mean = x / n
    std = (x2 / n - mean ** 2) ** 0.5
    
    print(
        f'Empty lines: {n_empty}\n'
        f'Count: {n}\n'
        f'Total tokens: {x}\n'
        f'Mean Tokens/Line: {mean}\n'
        f'Std Tokens/Line: {std}\n'
        f'Min Tokens/Line: {m}\n'
        f'Max Tokens/Line: {M}\n'
    )


if __name__ == '__main__':
    main()
