import sys

from train import main


if __name__ == "__main__":
    main(["--model", "lenet5", "--lr", "0.05"] + sys.argv[1:])
