import os
from sunpy.net import Fido, attrs as a
import astropy.units as u
import datetime
import os
import pdb
import argparse
import requests

_AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 335]

def _downloadFile(url, target):
    fileName = url.split("/")[-1]
    filePath = os.path.join(target, fileName)
    print("    Descargando %s..." % fileName, end="", flush=True)
    try:
        r = requests.get(url, stream=True, timeout=(10, 120))
        r.raise_for_status()  # lanza un error si el servidor respondió mal (ej. 404)
        with open(filePath, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        sizeMB = os.path.getsize(filePath) / (1024 * 1024)
        print(" OK (%.1f MB)" % sizeMB, flush=True)
        return True
    except Exception as e:
        print(" Fail: %s" % e, flush=True)
        return False

def pullData(targetBase, dt, account, wavelengths=_AIA_WAVELENGTHS):
    ts = dt.strftime("%Y%m%d %H:%M")
    print(ts, flush=True)
    tsPath = dt.strftime("%Y%m%d_%H%M")

    target = os.path.join(targetBase, tsPath)
    if not os.path.exists(target):
        os.mkdir(target)

    nDownloadedTotal = 0
    nExpectedTotal = 0
    for wl in wavelengths:
        print("Searching for %s -- %d A" % (ts, wl), flush=True)
        search = Fido.search(a.Time(dt, dt), a.jsoc.Series("aia.lev1_euv_12s"),
                              a.jsoc.Notify(account), a.Wavelength(wl * u.angstrom))

        if len(search[0]) == 0:
            print("  -> It didn't found files for %d A" % wl, flush=True)
            continue

        responses = [r for r in search[0]]
        resp = responses[0].table.client.request_data(responses[0].table)
        resp.wait()

        urls = resp.urls['url']
        print("  JSOC returns %d file(s) for %d A" % (len(urls), wl), flush=True)

        for url in urls:
            if url.endswith("spikes.fits"):
                continue
            nExpectedTotal += 1
            if _downloadFile(url, target):
                nDownloadedTotal += 1

    print("Download finished: %d/%d files saved as %s" % (nDownloadedTotal, nExpectedTotal, target), flush=True)

    # the way to do it without filtering
    #filen = Fido.fetch(search,path="./", max_conn=1)

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="directory to stage files")
    parser.add_argument("ts", help="date/time (example: 20240520_1200)")
    parser.add_argument("email", help="jsoc account email")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    target = args.target
    dt = datetime.datetime.strptime(args.ts, "%Y%m%d_%H%M")
    if not os.path.exists(target):
        os.mkdir(target)
    pullData(target, dt, args.email)