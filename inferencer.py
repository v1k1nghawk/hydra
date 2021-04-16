#!/usr/bin/env python3
########################################################################################################################
# Purpose: determination of two-spiked curve, approximated by the probability density function of the Gaussian Mixture
# Programmer: Andrew Art
########################################################################################################################

import signal
from sys import argv
from funcs import *
from version import *



# Handle interrupt signal
signal.signal(signal.SIGINT, signal.default_int_handler)

if __name__ == '__main__':

    print("Hydra ver" + str(VERSION_MAJOR) + "." + str(VERSION_MINOR))
    try:
        main(sys.argv, len(sys.argv))
    except KeyboardInterrupt:
        print("Script " + sys.argv[0] + " has been terminated")
    except Exception as e:
        print("ERROR: <" + str(e) + ">")
    else:
        print("OK")
    finally:
        print("Done.")

