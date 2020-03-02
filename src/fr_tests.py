from fractions import Fraction


def main():
    f1 = Fraction(5, 8)
    print(f1)
    num = f1.numerator
    den = f1.denominator
    f2 = Fraction(1, den)
    f3 = Fraction(num - 1, den)
    print(f2)
    print(str(f3))


if __name__ == '__main__':
    main()
