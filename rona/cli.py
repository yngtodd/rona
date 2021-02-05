"""Command line interface for rona."""
import argh


def greet() -> None:
    r"""Say hello, rona"""
    print(f'Hello, world!')


def main():
    parser = argh.ArghParser()
    parser.add_commands([greet])
    parser.dispatch()


if __name__=='__main__':
    main()
