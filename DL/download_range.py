"""
python download_range.py (save_path) (init_date) (end_date) (num_ts) jsocemail@example.com

"""

import argparse
import datetime
import random
from dlAIA_v2 import pullData # We reuse the function that already downloads a single instant in time


def generateRandomDates(start, end, n, seed=None):
    if n < 1:
        raise ValueError("n must be an integer greater than or equal to 1")

    rng = random.Random(seed)
    totalDurationSec = (end - start).total_seconds()

    dates = [
        start + datetime.timedelta(seconds=rng.uniform(0, totalDurationSec))
        for _ in range(n)
    ]
    dates = [roundTo12Seconds(dt) for dt in dates]
    dates.sort()  # order them chronologically, just for neatness
    return dates

def roundTo12Seconds(dt):
    dt = dt.replace(microsecond=0)
    roundedSeconds = round(dt.second / 12) * 12
    dt = dt.replace(second=0) + datetime.timedelta(seconds=roundedSeconds)
    return dt

def parseArgs():
    parser = argparse.ArgumentParser(
        description="Download n AIA files at random instants between two dates"
    )
    parser.add_argument("target", help="directory where files are saved")
    parser.add_argument("start", help="start date/time, format YYYYMMDD_HHMM")
    parser.add_argument("end", help="end date/time, format YYYYMMDD_HHMM")
    parser.add_argument("n", type=int, help="number of files to download")
    parser.add_argument("email", help="email of the JSOC account")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="optional seed so that random dates are reproducible"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()

    date_format = "%Y%m%d_%H%M"
    start_date = datetime.datetime.strptime(args.start, date_format)
    end_date = datetime.datetime.strptime(args.end, date_format)

    if end_date < start_date:
        raise ValueError("The end date must be posterior (or equal) to the start date")

    dates = generateRandomDates(start_date, end_date, args.n, seed=args.seed)

    print("Will download %d files at the following random dates:" % args.n)
    for dt in dates:
        print("  ", dt.strftime(date_format))

    for i, dt in enumerate(dates):
        print("\n[%d/%d] Downloading %s ..." % (i + 1, args.n, dt.strftime(date_format)))
        try:
            pullData(args.target, dt, args.email)
        except Exception as e:
            # If a date fails (e.g. no data available at that exact instant),
            # we report it and continue with the others
            print("  -> Error downloading %s: %s" % (dt.strftime(date_format), e))

    print("\nProcess completed.")