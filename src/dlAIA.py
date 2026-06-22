import os
from sunpy.net import Fido, attrs as a
import astropy.units as u
import datetime
import os
import pdb
import argparse


def pullData(targetBase, dt, account):
    ts = dt.strftime("%Y%m%d %H:%M")
    print(ts)
    tsPath = dt.strftime("%Y%m%d_%H%M")

    target = os.path.join(targetBase, tsPath)
    if not os.path.exists(target):
        os.mkdir(target)

    print("Searching for %s" % ts)
    search = Fido.search(a.Time(dt,dt), a.jsoc.Series("aia.lev1_euv_12s"), a.jsoc.Notify(account))

    # get all the urls, but only download the data (and not the spiking results)
    responses = [r for r in search[0]]
    resp = responses[0].table.client.request_data(responses[0].table)
    resp.wait()
    for i in range(len(resp.urls['url'])):
        url = resp.urls['url'][i]
        if url.endswith("spikes.fits"):
            continue
        # not exactly secure but good enough
	    # make sure you have wget installed
        os.system("cd %s && wget %s" % (target, url))

    # the way to do it without filtering
    #filen = Fido.fetch(search,path="./", max_conn=1)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="directory to stage files")
    parser.add_argument("ts", help="date/time to download in %%Y%%m%%d_%%H%%M format (20240520_1200)")
    parser.add_argument("email", help="jsoc account email")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    target = args.target
    dt = datetime.datetime.strptime(args.ts, "%Y%m%d_%H%M")
    if not os.path.exists(target):
        os.mkdir(target)
    pullData(target, dt, args.email)


