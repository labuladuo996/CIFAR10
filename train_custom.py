import sys

from train import main


if __name__ == "__main__":
    main(["--model", "custom"] + sys.argv[1:])
