import sys

from train import main


if __name__ == "__main__":
    main(["--model", "googlenet"] + sys.argv[1:])
