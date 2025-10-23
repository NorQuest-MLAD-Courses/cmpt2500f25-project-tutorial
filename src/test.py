from argparse import ArgumentParser

def foo(a, b):
    return (5 * a) ** b

if __name__ == "__main__":
    parser = ArgumentParser(description="blah")
    parser.add_argument("--a", type=int)
    parser.add_argument("--b", type=float, default=1.)
    args = parser.parse_args()
    print(foo(args.a, args.b))