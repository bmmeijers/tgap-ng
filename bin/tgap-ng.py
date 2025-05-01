#!/usr/bin/env python3

from tgap_ng.tgap import main
import datetime

if __name__ == "__main__":
    print("hi there, I am going to get you a tgap")
    print('start: {}'.format(datetime.datetime.now()))
    main()
    print('end: {}'.format(datetime.datetime.now()))
