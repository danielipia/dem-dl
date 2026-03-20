import numpy as np
import aiapy.calibrate
import scipy.optimize
from aiapy.calibrate import register, update_pointing, degradation, estimate_error
from aiapy.calibrate.utils import get_pointing_table, get_correction_table, get_error_table
from aiapy.psf import deconvolve, psf
from xrtpy.image_correction import remove_lightleak
from xrtpy.image_correction import deconvolve as xrt_deconvolve
from numcodecs import Blosc, BitRound
import xrtpy.response
import sunpy.map
import astropy
import astropy.units as u
import astropy.io.fits as fits
import pdb
import scipy.io
import scipy.ndimage
import matplotlib.pyplot as plt
import os
import time
import datetime
import highspy
import multiprocessing
import argparse
import sys
try:
    import gurobipy as gp
    from gurobipy import GRB
except:
    print("Couldn't import Gurobi; can't do a QP")


from scipy.sparse import csc_matrix

# AIA passbands and characteristic logTs for EUV data that are optically thin 
_aiaChan = ["94 A", "131 A", "171 A", "193 A", "211 A", "335 A"]
_aiaLogT = [("6.8",), ("7.0", "7.2"), ("5.8",), ("6.1", "7.3*"), ("6.3",), ("6.4",)]
_minimumDNs = 0.5 # minimum DN/s to be assumed throughout


def setupPage(target, logT=None, showStd=False, AIACubeCount=6):
    fh = open(os.path.join(target, "vis.htm"), "w")

    fh.write("<html><head>DEM Diagnostics</head><body>")
    fh.write("<h2>AIA Resynthesis</h2>")
    fh.write("<table>")
    fh.write("<tr><td></td><td>Measured</td><td>Resynthesis</td><td>Difference</td><td>Joint PDF</td></tr>")

    # show the AIA passbands in logT order (ascending) 
    reorder = [2, 3, 4, 5, 0, 1]

    if AIACubeCount > 6:
        for ii, i in enumerate(range(6,AIACubeCount)):
            reorder.append(i)
            _aiaChan.append("XRT %d" % ii)
            _aiaLogT.append(("High",))

    for i in reorder:
        fh.write("<tr>")
        fh.write("<td>Band: %s<br/>Char LogT: %s</td>" % (_aiaChan[i], ",".join(map(str, _aiaLogT[i]))))
        fh.write("<td><img src='aia_%d_meas.png' height=400></td>" % i)
        fh.write("<td><img src='aia_%d_synth.png' height=400></td>" % i)
        fh.write("<td><img src='aia_%d_synth_diff.png' height=400></td>" % i)
        fh.write("<td><img src='aia_%d_synth_jpdf.png' height=400></td>" % i)
    fh.write("</table>")

    fh.write("<h2>DEM Tables</h2>")

    fh.write("<table><tr><td></td><td>DEM</td></tr>")

    fh.write("<tr><td>Mean logT</td><td><img src='mean_logt.png' height=300></td></tr>") 
    fh.write("<tr><td>Std logT</td><td><img src='std_logt.png' height=300></td></tr>") 

    for i in range(logT.size):
        if logT is None:
            fh.write("<tr><td>Bin %d</td>" %  i)
        else:
            fh.write("<tr><td>Bin %d<br/>LogT %.1f</td>" %  (i,logT[i]))
        fh.write("<td><img src='%d_vis.png' height=300></td>" % i) 
        if showStd:
            fh.write("<td><img src='%d_nzstd.png' height=300></td>" % i) 
            fh.write("<td><img src='%d_pz.png' height=300></td>" % i) 
        fh.write("</tr>")

    fh.write("</table>")
    fh.write("</body>")
    fh.write("</html>")
    fh.close()


def getBasis(R, logT, alphas=[0.0, 0.1, 0.2, 0.6], notrunc=False):
    """Make an expanded basis set"""
    nBins = R.shape[1]
    nAlphas = len(alphas)
    assert(nBins == logT.size)

    basis = np.zeros((nBins, nBins*nAlphas))
    for ai in range(nBins):
        basis[ai,ai] = 1.0

    for ai in range(1, nAlphas):
        a = alphas[ai]
        for bi in range(nBins):
            refValue = logT[bi]
            col = ai*nBins + bi
            diffLogT = logT - logT[bi]
            basisResult = np.exp(-(diffLogT**2 / a**2))
            # apply truncation: skip for bins 0-17 when notrunc=True, always apply to extended bins
            if notrunc and bi < 18:
                pass  # skip truncation for original aia bins
            else:
                basisResult[basisResult<0.04] = 0
            basis[:,col] = basisResult

    return basis


def nnInterpNaN(X):
    """Given HxWxC image X, nearest neighbor interpolate all pixels with a nan
    in any channel. Not very efficient"""
    M = np.any(np.isnan(X),axis=0)

    distanceIndMulti = scipy.ndimage.distance_transform_edt(M, return_distances=False, return_indices=True)
    distanceInd = np.ravel_multi_index(distanceIndMulti, M.shape)

    X2 = X.copy()
    for c in range(X.shape[0]):
        Xc = X[c,:,:]
        X2[c,:,:] = Xc.ravel()[distanceInd]
    return X2 

def dumpSynthesis(target, name, AIACube, DEMCube, R, wavelengths):
    """
    Resynthesize the AIA Data, compared to the AIA data

    target: where to dump the data
    name: identifier to append in the pngs
    AIACube: C x H x W
    DEMCube: nBins x H x W
    R: response function (C x nBins)
    wavelengths: the corresponding wavelength (in A), for the right colormap
    """
    resynth = np.zeros(AIACube.shape)
    _, H, W = AIACube.shape
    for i in range(H):
        for j in range(W):
            resynth[:,i,j] = R@DEMCube[:,i,j]

    for i in range(AIACube.shape[0]):
        if i < 6:
            cm = plt.get_cmap("sdoaia%d" % wavelengths[i])
        else:
            cm = plt.get_cmap("inferno")

        plt.imsave(os.path.join(target, "aia_%d_%s.png" % (i, name)), 
                    np.maximum(0,resynth[i,:,:])**0.5,
                    vmin=0, vmax=np.nanmax(AIACube[i,:,:])**0.5, cmap=cm)
        plt.imsave(os.path.join(target, "aia_%d_meas.png" % i), 
                    np.maximum(0,AIACube[i,:,:])**0.5,
                    vmin=0, vmax=np.nanmax(AIACube[i,:,:])**0.5, cmap=cm)
        
        #plot a difference map
        vrange = np.nanmax(AIACube[i,:,:])*0.2
        diff = resynth[i,:,:] - AIACube[i,:,:]
        plt.imsave(os.path.join(target, "aia_%d_%s_diff.png" % (i, name)), 
                    diff, vmin=-vrange, vmax=vrange, cmap='bwr')


        x = np.maximum(AIACube[i,:,:].reshape(-1), 0.5)
        y = np.maximum(resynth[i,:,:].reshape(-1), 0.5)
       
        k = ~np.isnan(x) 
        clip = np.maximum(np.nanmax(np.log10(x[k])), np.nanmax(np.log10(y[k])))

        plt.figure(figsize=(4,4))
        plt.hexbin(x[k], y[k], xscale='log', yscale='log', extent=[1,clip,1,clip], bins='log', gridsize=200)
        plt.plot([10**1, 10**clip],[10**1, 10**clip], c='k', linestyle='--')
        plt.axis('square')
        plt.savefig(os.path.join(target, "aia_%d_%s_jpdf.png" % (i, name)), dpi=300)
        plt.close()


def dumpDiagnostics(target, DEMCube, logT):
    """
    Plot some diagnostic information about the DEM cube, including:
    - mean of the logT distribution
    - std of the logT distribution
    """
    nLogT = DEMCube.shape[0]
    logTRange = np.max(logT) - np.min(logT)

    assert(nLogT == logT.size)
    # convert to a distribution, but some might be all 0, so add 1e-8 to stabilize
    DEMDistr = DEMCube / (1e-8 + np.sum(DEMCube, axis=0, keepdims=True))
    meanLogT = np.sum(logT.reshape(-1,1,1) * DEMDistr, axis=0, keepdims=True)
    stdLogT = np.sum((logT.reshape(-1,1,1) - meanLogT)**2 * DEMDistr, axis=0, keepdims=True)**0.5

    plt.imsave(os.path.join(target, "mean_logt.png"), meanLogT[0,:,:], vmin=5.5, vmax=8.0, cmap='inferno')
    plt.imsave(os.path.join(target, "std_logt.png"), stdLogT[0,:,:], vmin=0, vmax=0.5*2.5, cmap='inferno')

def solveQP(t):
    """Solve a quadratic program with Gurobi

    argmin_x 1/2 * lam x^T I x + 1^T x s.t. lb <= Dx <= ub, x>=0
    
    where lam is a parameter

    which is just

    argmin_x 1/2 lam ||x||^2 + ||x||1 s.t. lb <= Dx <= ub, x>=0
    """
    
    D, meas, lb, ub, args = t

    ncol = D.shape[1]
    c = np.ones((D.shape[1],))
    Q = np.eye(ncol) * 0.5 * args.fitqpscale

    # build a completely silent environment
    env = gp.Env(empty=True)
    env.setParam('LogToConsole',   0)
    env.setParam('OutputFlag',     0)
    env.setParam('Presolve',       0)
    env.setParam('LogFile',      '/dev/null')
    env.start()
    
    # now build the model in that env
    m = gp.Model("dem", env=env)
    x = m.addMVar(name="x", shape=(ncol,), lb=0.0)
    m.setObjective(c @ x + x.T @ Q @ x, GRB.MINIMIZE)
    m.addConstr(D @ x <= ub)
    m.addConstr(D @ x >= lb)
    m.optimize()
    
    # return the results only if it's optimized
    if m.Status == GRB.OPTIMAL:
        return x.X
    else:
        return None


def solveQPWithFit(t):
    """Solve a quadratic program with Gurobi

    This combines goodness of fit, plus hard constraints, plus regularization
    
    argmin_x
        1/(2n) * ||Rx-o||_2^2 + 
        (al)*||x||_1 
        + (1/2)a(1-l)*||x||_2^2

    s.t.  x >= 0, lb <= Dx <= ub

    which can be written as

    argmin_x
        x^T Q x + c^T x 
    with 
        Q = 1/(2n) * I + a(1-l)/2 R^T R
        c = al 1 + 1/(2n) R^To

    s.t. x >= 0, lb <= Dx <= ub
    """
    D, meas, lb, ub, args = t

    # 
    n = D.shape[0]
    fitTermScale = 1.0 / (2.0 * n)
    a = args.fitlinearalpha
    l = args.fitlinearl1ratio

    # compute the version that rescales according to noise to solve 
    # make the fit actually look like a chi-square
    tol = np.maximum(meas - lb, _minimumDNs)
    Du = D / tol[:,None]
    measu = meas / tol

    ncol = D.shape[1]
    Q = np.eye(ncol) * a * (1-l) / 2
    Q += Du.T@Du * fitTermScale

    c = np.ones((D.shape[1],)) * a * l
    c += -2*Du.T@measu * fitTermScale

    m = gp.Model("dem")
    m.Params.Presolve = 0
    m.Params.LogToConsole = 0
    m.Params.OutputFlag = 0
    x = m.addMVar(name="x", shape=(ncol,), lb=0.0)
    m.setObjective(c@x + x.T@Q@x, GRB.MINIMIZE)
    m.addConstr(D@x <= ub)
    m.addConstr(D@x >= lb)
    m.optimize()
    # return the results only if it's optimized
    if m.Status == GRB.OPTIMAL:
        return x.X
    else:
        return None


def solveLP(t):
    """Solve a linear program, skipping scipy.

    argmin_x 1^T x  s.t., lb <= Dx <= ub, x>=0

    D, meas, lb, ub are passed in a single tuple to help with multiprocessing

    This is considerably faster because a lot of scipy is handling general 
    cases, and we're calling the solve repeatedly
    """
    D, meas, lb, ub, _ = t

    # create inequality matrices and vectors, saving as a sparse matrix
    Aineq = csc_matrix(D)

    c = np.ones((D.shape[1],))

    # setup environment
    highs = highspy._Highs()
    highs.setOptionValue("log_to_console", False)
    highs_options = highspy.HighsOptions()
    setattr(highs_options, 'log_to_console', False)

    # set up the lp
    lp = highspy.HighsLp()
    numcol, numrow = c.size, lb.size
    lp.num_col_ = numcol
    lp.num_row_ = numrow
    lp.a_matrix_.num_col_ = numcol
    lp.a_matrix_.num_row_ = numrow
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise

    # setup the cost
    lp.col_cost_ = c

    # constraints on x: within [0, \inf]
    lp.col_lower_ = np.zeros_like(c)
    lp.col_upper_ = np.ones_like(c) * np.inf

    # constraints on D@x: within [lb, ub]
    lp.row_lower_ = lb
    lp.row_upper_ = ub


    # put in inequality here
    lp.a_matrix_.start_ = Aineq.indptr
    lp.a_matrix_.index_ = Aineq.indices
    lp.a_matrix_.value_ = Aineq.data

    # run and return either the results if successful or None otherwise
    highs.passOptions(highs_options)
    highs.passModel(lp)
    run_status = highs.run()

    status = highs.getModelStatus()

    if status == highspy.HighsModelStatus.kOptimal:
        return np.array(highs.getSolution().col_value)
    else:
        return None

def solveLPRefit(t):
    """
    Fit the data by first solving a LP (the BP method), and then re-fitting on
    the selected coefficients using non-negative least-squares. This is
    experimental, but seems to reduce underestimation
    """
    D, meas, lb, ub, _ = t
    lpSol = solveLP(t)
    if lpSol is None:
        return None
    newSol = np.zeros(lpSol.shape)
    nzBin = (lpSol > 0).ravel()
  
    # tol is proportional to sigma. Argmin_x>=0 ||Dx/sigma - y/sigma||^2 
    # properly accounts for the noise model
    tol = meas - lb
    try:
        # fit a non-negative least squares
        newSol[nzBin], _ = scipy.optimize.nnls(D[:,nzBin] / tol[:,None], meas / tol)
    except:
        # in a few rare cases, things fail to converge, so return the LP solution
        return lpSol
        
    return newSol

def solveQPRefit(t):
    """
    Fit the data by first solving a QP, and then re-fitting on
    the selected coefficients using non-negative least-squares. This is
    experimental, but seems to reduce underestimation
    """
    D, meas, lb, ub, _ = t
    qpSol = solveQP(t)
    if qpSol is None:
        return None
    newSol = np.zeros(qpSol.shape)
    nzBin = (qpSol > 0).ravel()
  
    # tol is proportional to the noise sigma: 
    #Argmin_x>=0 ||Dx/sigma - y/sigma||^2 properly accounts for the noise model
    tol = meas - lb
    try:
        # fit a non-negative least squares
        newSol[nzBin], _ = scipy.optimize.nnls(D[:,nzBin] / tol[:,None], meas / tol)
    except:
        # in a few rare cases, things fail to converge, so return the LP solution
        return qpSol
        
    return newSol

def solveLasso(t):
    """
    Solve a lasso problem
        argmin_{x>=0} 1/(2N) ||Dx - y||^2 + a * ||x||_1
    where a is args.fitlinearalpha
    """
    from sklearn import linear_model 
    D, meas, lb, ub, args = t
    # sklearn seems more sensitive
    tol = np.maximum(meas - lb, _minimumDNs)
    m = linear_model.Lasso(alpha=args.fitlinearalpha, 
                            fit_intercept=False, max_iter=10000, positive=True)
    try:
        m.fit(D / tol[:,None], meas / tol)
        return m.coef_
    except:
        return None


def solveElasticNet(t):
    """
    Solve an elastic net problem
        argmin_{x>=0} 1/(2N) ||Dx - y||^2 + a (l) * ||x||_1 + a (1-l) 0.5 * ||x||^2
    where a is args.fitlinearalpha and l is args.fitlinearl1ratio
    """
    from sklearn import linear_model 

    D, meas, lb, ub, args = t
    tol = np.maximum(meas - lb, _minimumDNs)
    m = linear_model.ElasticNet(alpha=args.fitlinearalpha, l1_ratio=args.fitlinearl1ratio, 
                                fit_intercept=False, max_iter=10000, positive=True)
    try:
        m.fit(D / tol[:,None], meas / tol)
        return m.coef_
    except:
        return None

_solvers = {
    "lp": solveLP,
    "qp": solveQP,
    "lp_nnls": solveLPRefit,
    "qp_nnls": solveQPRefit,
    "qp_fit": solveQPWithFit,
    "lasso": solveLasso,
    "elasticnet": solveElasticNet,
}

def invertDEMCube(AIACube, AIAErrors, RData, args):
    """
    Given:
    - AIACube: CxHxW cube of AIA data
    - AIAErrors: CxHxW cube of errors 
    - RData: dictionary containing response + basis matrices
    - args the argparse function; it's just easier to pass this here
    """
    # Given response matrix and basis matrix in RData
    R, B = RData['R'], RData['B']

    doParallel = args.parallel
    if doParallel < 0:
        doParallel = multiprocessing.cpu_count()

    solveFn = _solvers[args.fitfn]

    # get shapes
    C, H, W = AIACube.shape
    DEMC = np.nan * np.ones((R.shape[1], H, W))

    # total matrix is R (nObs x nLogTBins) @ B (nLogTBins x nBasis)
    D = R@B

    tic = time.time()
    # create indices
    Y, X = np.meshgrid(np.arange(0, H), np.arange(0, W), indexing='ij')
    Y, X = Y.ravel(), X.ravel()
    
    AIACubeFlat = AIACube.reshape(AIACube.shape[0],-1).T # N Pixels x 6
    # Tolerance is errors * tolfac (e.g., 1.4 * std)
    tol = AIAErrors.reshape(AIAErrors.shape[0],-1).T * args.tolfac # N Pixels x 6
    tol[np.isnan(tol)] = 0.1

    # given inds, and a scale, create a package for solveLP to handle that
    # contains D (R@B), and then the lower bound (measurement - scale*error) 
    # and the lower bound (measurement + scale*error)
    packForMap = lambda inds, scale: ((D, AIACubeFlat[i,:], AIACubeFlat[i,:]-scale*tol[i,:], AIACubeFlat[i,:]+scale*tol[i,:], args)  for i in inds)

    #first, solve every pixel
    inds = [(Y[i], X[i]) for i in range(Y.size)]
    work = packForMap(range(Y.size), 1.0)
    toc = time.time()
    print("Setup took %.2f" % (toc-tic))

    tic = time.time()
    if doParallel != 0:
        P = multiprocessing.Pool(doParallel)
        results = P.map(solveFn, work)
    else:
        results = [solveFn(t) for t in work]


    if not args.zerochill:
        # find the indices that weren't solved, and create new problems with looser
        # tolerances
        invalid = [i for i in range(len(inds)) if results[i] is None]
        print("Cleaning up %d pixels" % len(invalid))
        work2 = packForMap(invalid, 3.0)
        if doParallel != 0:
            resultsInvalid = P.map(solveFn, work2)
        else:
            resultsInvalid = [solveFn(t) for t in work2]
            
        # find the indices that weren't solved in the second round, and create even
        # looser tolerances
        invalid2 = [invalid[ii] for ii in range(len(resultsInvalid)) if resultsInvalid[ii] is None]
        print("Cleaning up %d pixels" % len(invalid2))
        work3 = packForMap(invalid2, 5.0)
        if doParallel != 0:
            resultsInvalid2 = P.map(solveFn, work3)
            P.close()
        else:
            resultsInvalid2 = [solveFn(t) for t in work3]

    else:
        print("Skipping relaxation step")
        resultsInvalid, resultsInvalid2 = [], []

    toc = time.time()
    print("BP took %.2f" % (toc-tic))

    # load the data in 
    tic = time.time()
    for ii, (i, j) in enumerate(inds):
        if results[ii] is not None:
            DEMC[:,i,j] = B@results[ii]

    for ii in range(len(resultsInvalid)):
        if resultsInvalid[ii] is not None:
            i, j = inds[invalid[ii]]
            DEMC[:,i,j] = B@resultsInvalid[ii]

    for ii in range(len(resultsInvalid2)):
        if resultsInvalid2[ii] is not None:
            i, j = inds[invalid2[ii]]
            DEMC[:,i,j] = B@resultsInvalid2[ii]

    toc = time.time()
    print("Repacking took %.2f" % (toc-tic))
    return DEMC


def fitAffine(XYXPYP):
    """
    Given a Nx4 matrix of [x,y,x',y'], fit an affine transformation
    between the points via least-squares
    """
    N = XYXPYP.shape[0]
    A = np.zeros((2*N,6))
    b = np.zeros((2*N,1))
    for i in range(N):
        # design matrix
        A[2*i,:3] = XYXPYP[i,0], XYXPYP[i,1], 1
        A[2*i+1,3:] = XYXPYP[i,0], XYXPYP[i,1], 1
        # target vector
        b[2*i,0] = XYXPYP[i,2]; b[2*i+1,0] = XYXPYP[i,3]

    # results are in A \ b
    model,_,_,_ = np.linalg.lstsq(A,b,rcond=None)
    return model

def mapXRTToAIA(XRT, AIA):
    """
    Given an XRT Map and AIA Map, return the XRT data mapped onto the AIA
    coordinate system
    """
    XRTH, XRTW = XRT.data.shape[0], XRT.data.shape[1]
    AIAH, AIAW = AIA.data.shape[0], AIA.data.shape[1]

    X, Y = np.meshgrid(np.arange(0,AIAH,4), np.arange(0,AIAW,4))
    Xp, Yp = astropy.wcs.utils.pixel_to_pixel(AIA.wcs, XRT.wcs, X, Y)
    
    XYXPYP = np.hstack([Y.reshape(-1,1), X.reshape(-1,1), Yp.reshape(-1,1), Xp.reshape(-1,1)])
    k = np.all(~np.isnan(XYXPYP), axis=1)
    A = fitAffine(XYXPYP[k,:])
    A = np.vstack([A.reshape((2,3)), np.array([[0,0,1]])])

    res = scipy.ndimage.affine_transform(XRT.data, A, output_shape=(AIAH,AIAW), order=1, cval=np.nan)
    return res


def getXRTActiveFilter(XRT):
    """
    Given an XRT map, return the filter name (e.g., Be_thin)
    """
    open1 = XRT.meta["ec_fw1_"].lower() == "open"
    open2 = XRT.meta["ec_fw2_"].lower() == "open"

    assert open1 != open2, "Exactly one filter has to be open"

    return XRT.meta["ec_fw2_"].strip() if open1 else XRT.meta["ec_fw1_"].strip()


def getTimeFromXRTFilename(fn):
    """
    Get a datetime from an XRT filename
    """
    return datetime.datetime.strptime(fn.split(".")[0], "comp_XRT%Y%m%d_%H%M%S")

def getMedianDatetime(dts):
    """
    Get the mediod datetime
    """
    bestInd, bestDistance = None, None
    for i in range(len(dts)):
        
        totalDistance = 0
        for j in range(len(dts)):
            totalDistance += abs((dts[i] - dts[j]).total_seconds())
        
        if (bestDistance is None) or (totalDistance < bestDistance):
            bestInd, bestDistance = i, totalDistance
    return dts[bestInd]


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", help="source folder with AIA data")
    parser.add_argument("target", help="target file")
    parser.add_argument("--xrt", default=False, action="store_true", help="scan for XRT data and use it if possible")
    parser.add_argument("--extendto8", default=False, action="store_true", help="extend logT bins up to 8.0 with zero response (effectively AIA-only for extended bins)")
    parser.add_argument("--zerochill", default=False, action="store_true", help="don't do successive relaxations of the tolerances")
    parser.add_argument("--corr_table", default="aia_corr.csv", help="correction table to use for degradation factors (csv is better)")
    parser.add_argument("--pointing_file", default="", help="cached pointing table file (ecsv format). if provided, filters this instead of querying jsoc")

    parser.add_argument("--visTarget", default="", help="visualizaton target")
    parser.add_argument("--noisy", default=0, type=int, help="generate single noisy realization (integer for seed offset)")
    parser.add_argument("--noisescale", default=0.5, type=float, help="scale factor for noise (1.0 = full noise, 0.5 = half, etc.)")
    # Whether to use the simple phton noise estimate or a full model
    parser.add_argument("--errorfn", default="photon", help="error estimation function (photon, full)")
    # We constrain the problem to [meas - tolfac*error(meas), meas + tolfac*error(meas)]
    parser.add_argument("--tolfac", default=1.4, type=float, help="how much to multiply the error by to produce constraints")
    parser.add_argument("--deconvolve", default=False, action="store_true", help="whether to deconvolve the AIA data first")

    # The Mark Cheung version reports also using a basis function with
    # alpha=0.6. I've found that in some cases, this has a lot of contamination
    # since the basis function is wide. So optimization will deposit a moderate
    # contribution in a more moderate temperature (logT < 7), but then the
    # wings of the basis will dump tons of plasma with logT > 7
    parser.add_argument("--basisAlphas", default="0.0_0.1_0.2", help="Alphas for basis as _ separated floats")
    parser.add_argument("--notrunc", default=False, action="store_true", help="disable basis function truncation")

    # Fit function that finds a DEM
    parser.add_argument("--fitfn", default="lp", help="Which function to use to fit (Options: %s)" % ",".join(sorted(_solvers.keys())))
    # Fit arguments
    parser.add_argument("--fitqpscale", default=1, type=float, help="Scale of quadratic function for QP")
    parser.add_argument("--fitlinearalpha", default=1, type=float, help="Lasso/Elastic Net regularization strength")
    parser.add_argument("--fitlinearl1ratio", default=0.5, type=float, help="Elastic Net L1 Ratio")

    # Subsample arugments
    parser.add_argument("--decimate", type=int, default=1, help="decimation factor")
    parser.add_argument("--crop", default="", help="crop (sy,sx,h,w)")
    parser.add_argument("--parallel", type=int, default=-1, help="parallel (0 = none, -1 = number of cores, >0 = user specified)")
    return parser.parse_args()


if __name__ == "__main__":
    # The response function is stored near 1e-28 to be in the proper units, and
    # so we need to scale things to a range that makes sense numerically
    scale = 10**26
    wavelengths = [94, 131, 171, 193, 211, 335]

    args = parseArgs()
    assert(4096 % args.decimate == 0)
    if args.xrt:
        assert args.zerochill, "XRT errors are currently iffy; iterative relaxation should be disabled with --zerochill"

    # grab a file in a folder by the nominal wavelength
    getByWL = lambda p, wl: [fn for fn in os.listdir(p) if fn.endswith("%d.image_lev1.fits" % wl)][0]
    # get all the files in a directory
    AIAFiles = [os.path.join(args.src, getByWL(args.src, wl)) for wl in wavelengths]


    # load the R data
    RData = np.load("RData.npz")
    R, logT = RData['R'], RData['logT']


    # Load the data
    AIAMaps = [sunpy.map.Map(fn) for fn in AIAFiles]
    AIAFits = [fits.open(fn) for fn in AIAFiles]


    ###########################################################################
    # Get the information needed to properly scale the images. This needs to be
    # applied at the very end -- the noise model is mainly driven by photon
    # noise, and this needs the original counts (not counts per second)
    exposures = [f[1].header["EXPTIME"] for f in AIAFits]

    if os.path.isfile(args.corr_table):
        correction_table = get_correction_table(args.corr_table)  # local read
        print("Using correction table %s" % args.corr_table)
    else:
        print("Correction table %s not found, falling back to JSOC/SSW" % args.corr_table)
        correction_table = get_correction_table("JSOC")  # falls back to JSOC/SSW
        # cache for future use (astropy QTable)
        correction_table.write(args.corr_table, format='csv', overwrite=True)
        print("Using correction table from JSOC, cached to %s" % args.corr_table)

    degradationFactors = []
    for i in range(len(AIAMaps)):
        degradationFactors.append(degradation(wavelengths[i] * u.angstrom, AIAMaps[i].date, correction_table=correction_table))

    # the total scale factor (to divide by) is the exposure time and the
    # divisive degradataion factor
    scaleFactor = [(exposures[i] * degradationFactors[i]).to_value()[0] for i in range(len(exposures))]

    #############################################################################
    # Now handle loading the data, updating the pointing, potential
    # deconvolving, and then registering the images
    
    # use cached pointing table if provided, otherwise query jsoc
    if args.pointing_file and os.path.isfile(args.pointing_file):
        # load master table once and cache in function attribute
        if not hasattr(get_pointing_table, '_master_cache'):
            from astropy.table import QTable
            get_pointing_table._master_cache = QTable.read(args.pointing_file, format='ascii.ecsv')
            print(f"loaded master pointing table from {args.pointing_file}")
        
        # filter to needed time range (local operation, no network)
        t_start = AIAMaps[0].date - 12 * u.h
        t_end = AIAMaps[0].date + 12 * u.h
        mask = (get_pointing_table._master_cache['T_START'] >= t_start) & (get_pointing_table._master_cache['T_START'] <= t_end)
        pointing_table = get_pointing_table._master_cache[mask]
        print(f"using cached subset: {len(pointing_table)} entries for {AIAMaps[0].date}")
    else:
        # fallback to jsoc query (current behavior)
        pointing_table = get_pointing_table("JSOC", time_range=(AIAMaps[0].date - 12 * u.h, AIAMaps[0].date + 12 * u.h))

    if args.deconvolve:
        AIAMaps = [update_pointing(m, pointing_table=pointing_table) for m in AIAMaps]
        AIAMapsN = []
        for i in range(len(AIAMaps)):
            print("Computing PSF %d" % i)
            cpsf = psf(wavelengths[i]*u.angstrom)
            print("Deconvolving %d" % i)
            AIAMapsN.append(deconvolve(AIAMaps[i], psf=cpsf))
        AIAMaps = AIAMapsN
        AIAMaps = [register(m) for m in AIAMaps]
    else:
        # first geometrically register the images
        AIAMaps = [register(update_pointing(m, pointing_table=pointing_table)) for m in AIAMaps]


    # produce error maps, using one of two methods
    AIAErrors = []
    if args.errorfn == "full":
        # compute the error function using AIApy. This includes photon noise, 
        # read noise, etc.
        errorTable = get_error_table(source="SSW")
        for i in range(len(AIAMaps)):
            AIAErrors.append(estimate_error(
                np.maximum(AIAMaps[i].data, _minimumDNs) * u.DN / u.pix, wavelengths[i] * u.angstrom, 
                error_table=errorTable).value)

    elif args.errorfn == "photon":
        # This matches the Mark Cheung DEM method, but the tolfac is handled in
        # invertDEMCube, not here. This helps separate out tolfac from
        # estimates of the error
        for i in range(len(AIAMaps)):
            AIAErrors.append(np.maximum(AIAMaps[i].data, _minimumDNs)**0.5)
        
    else:
        print("Don't recognize error function")
        sys.exit(1)

    # Due to the register function, things might be slightly smaller than 4096^2
    # Just zero-pad to make them the same size
    AIACube = [AIAMap.data for AIAMap in AIAMaps]
    for i in range(len(AIACube)):
        if AIACube[i].shape[0] != 4096:
            diff = 4096 - AIACube[i].shape[0]
            AIACube[i] = np.pad(AIACube[i], diff//2)
            AIAErrors[i] = np.pad(AIAErrors[i], diff//2)

    # These are in the original measured DN/s, without a degradation correction
    # or exposure correction, and need to be divided by scaleFactor
    AIACube = np.concatenate([c[None,:,:] for c in AIACube], axis=0)
    AIAErrors = np.concatenate([c[None,:,:] for c in AIAErrors], axis=0)
    
    # create separate error array for noise resampling
    # for AIA, this is the same as AIAErrors
    AIAResampleError = AIAErrors.copy()

    #############################################################################
    # Extend logT to 8.0 ONLY if XRT is present
    # For AIA-only with --extendto8, we'll pad zeros at the end instead
    # This prevents the solver from putting values in zero-response bins
    
    #############################################################################
    # Now handle XRT data
    # The temperature response is a function of time and depends on what filter
    # we have, so we'll just add to R and then AIAFiles and AIAErrors
    #
    if args.xrt:
        XRTFileNames = [fn for fn in os.listdir(args.src) if fn.startswith("comp_XRT") and (not fn.endswith(".gmap.fits"))]

        # if we just downloaded all the data, we'll have a heap of files; make
        # sure we don't use multiple samples with the same filter
        byFilter = {}
        for fn in XRTFileNames:
            XRTMap = sunpy.map.Map(os.path.join(args.src, fn))
            filterName = getXRTActiveFilter(XRTMap)

            if filterName not in byFilter:
                byFilter[filterName] = []
            byFilter[filterName].append(fn)

        bannedFilters = ['Ti_poly']
        print("XRT filters before\n", byFilter)
        byFilter = {f:byFilter[f] for f in byFilter if f not in bannedFilters}
        print("XRT filters after\n", byFilter)

        # if we need to pick, pick as close as possible to the middle of the aia data 
        AIAFileTimes = [datetime.datetime.strptime(fn.split(".")[-4], "%Y-%m-%dT%H%M%SZ") for fn in AIAFiles]
        medianAIATime = getMedianDatetime(AIAFileTimes)
       
        XRTFileNames = []
        for filterName in byFilter:
            filterTimes = [getTimeFromXRTFilename(fn) for fn in byFilter[filterName]]
            filterDistance = [abs((dt-medianAIATime).total_seconds()) for dt in filterTimes]
            XRTFileNames.append(byFilter[filterName][np.argmin(filterDistance)])

        # now load and process
        XRTFiles = [os.path.join(args.src, fn) for fn in XRTFileNames]

        XRTMapsOrig = [sunpy.map.Map(fn) for fn in XRTFiles]

        XRTMaps = []
        for XRTMap in XRTMapsOrig:
            # try to remove lightleak; some filters don't seem to have this
            # calibration and so it fails, so move forward
            try:
                XRTMap = remove_lightleak(XRTMap)
            except ValueError:
                pass
            XRTMaps.append(XRTMap)

        if args.deconvolve:
            print("not deconvolving XRT data -- synoptic maps seem to blow up if this is done")
            # XRTMaps = [xrt_deconvolve.deconvolve(XRTMap) for XRTMap in XRTMaps]

        XRTRemappedData = [mapXRTToAIA(XRTMap, AIAMaps[0]) for XRTMap in XRTMaps]

        trfs = []

        # extend logT for XRT (whether or not --extendto8 was specified)
        logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
        if logT[-1] < 7.3:  # only extend if not already extended
            R = np.hstack([R, np.zeros((R.shape[0], logTExpand.size))])
            logT = np.hstack([logT, logTExpand])

        for i, XRTMap in enumerate(XRTMaps):

            # get the filter corresponding to the fits file 
            xrtFilter = getXRTActiveFilter(XRTMap)

            # create a temperature response, indexed along the chianti locations
            trf = xrtpy.response.TemperatureResponseFundamental(xrtFilter, XRTMap.meta['date_obs'])
            xrtLogT = np.log10(trf.CHIANTI_temperature.to_value()) 
            xrtResp = trf.temperature_response().to_value()

            # We apply a 3x correction factor. See Wright et al. ApJ 2017 
            # Microflare Heating of a Solar Active Region Observed with NuSTAR, 
            # Hinode/XRT, and SDO/AIA, which proposes 2 but says that the error
            # is ~2-3. If we do 2, we seem to get worse fits anecdotally
            newResp = np.interp(logT, xrtLogT, xrtResp) * 3

            # The error needs some work
            data = XRTRemappedData[i] 
            #error = np.maximum(100,data) * 0.2

            # if data is missing, set it to zero with ultrawide errors
            # this is not the best solution -- it should probably be a nominal corona
            # value if it's outside the full-disk XRT FOV
            # handle missing XRT data and create separate resample error
            # for missing pixels: use nearest neighbor interpolation for data,
            # and set large LP error (1e8) with zero resample error (no noise)
            missingXRT = np.isnan(data)
            
            # fill missing data with nearest neighbor
            nearestIndex = scipy.ndimage.distance_transform_edt(missingXRT, return_distances=False, return_indices=True)
            data = data[nearestIndex[0], nearestIndex[1]]
            
            # recompute error for filled pixels (now that data is not NaN)
            error = np.maximum(100, data) * 0.2
            
            # create resample error array (for noise generation)
            resample_error = error.copy() / 10  # 10x smaller for measured XRT pixels
            resample_error[missingXRT] = 0      # no noise for originally missing pixels
            
            # set LP bounds error (huge for missing pixels so LP effectively ignores them)
            error[missingXRT] = 10**8

            # add the data, errors, and response function
            AIACube = np.vstack([AIACube, data[None, :, :]])
            AIAErrors = np.vstack([AIAErrors, error[None, :, :]])
            AIAResampleError = np.vstack([AIAResampleError, resample_error[None, :, :]])

            R = np.vstack([R, newResp.reshape(1,-1)])
            wavelengths.append("xrt_"+xrtFilter)
            
            # The XRT errors are a bit messy, so just assume that we don't
            # scale things; but the exposure is nominally in
            # XRTMap.meta['exptime']). That said, the synoptic map is a HDR map
            # that is created from multiple exposures
            scaleFactor.append(1.0) 


    if args.crop != "":
        cropSy, cropSx, cropH, cropW = [int(v) for v in args.crop.split(",")]
        AIACube = AIACube[:, cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]
        AIAErrors = AIAErrors[:, cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]
        AIAResampleError = AIAResampleError[:, cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]

    # save original before any modifications (for --noisy)
    if args.noisy > 0:
        AIACubeOrig_noisy = AIACube.copy()
        AIAErrorsOrig_noisy = AIAErrors.copy()
        AIAResampleErrorOrig_noisy = AIAResampleError.copy()

    if args.decimate != 1:
        assert(AIACube.shape[1] % args.decimate == 0)
        AIACubeOrig = AIACube.copy()
        AIAErrorsOrig = AIAErrors.copy()
        AIAResampleErrorOrig = AIAResampleError.copy()

        AIACube = AIACube[:, ::args.decimate, ::args.decimate]
        AIAErrors = AIAErrors[:, ::args.decimate, ::args.decimate]
        AIAResampleError = AIAResampleError[:, ::args.decimate, ::args.decimate]

    # miscellaneous bookeeping that is here so that XRT code can work
    # convert scaleFactor to the right size
    scaleFactor = np.array(scaleFactor).reshape(-1, 1, 1)

    # apply noise if requested
    if args.noisy > 0:
        
        # extract timestamp from source folder (format: 20140102_062956)
        timestamp_str = os.path.basename(args.src.rstrip('/'))
        # include noisy offset in hash to avoid seed overflow when adding
        noise_offset = int(args.noisy)
        assert len(timestamp_str) == 15, "Source folder name should be a timestamp of the form YYYYMMDD_HHMMSS"
        simple_seed = int(timestamp_str + str(noise_offset)) % (2**32)

        print(f"applying gaussian noise with seed {simple_seed} (timestamp + offset: {int(timestamp_str + str(noise_offset))})")
        print(f"  noise scale: {args.noisescale} (1.0 = full noise)")
        print(f"  using AIAResampleError for noise (separate from AIAErrors for LP bounds)")

        np.random.seed(simple_seed)
        
        # generate gaussian noise using AIAResampleError (NOT AIAErrors!)
        epsilon = np.random.randn(*AIACube.shape)
        AIACube = AIACube + args.noisescale * AIAResampleError * epsilon

    # Generate the basis
    # parse the alphas for creating the solution
    basisAlphas = list(map(float, args.basisAlphas.split("_")))
    assert(all(a >= 0.0 for a in basisAlphas))
    R = (R * scale).astype(np.float64)
    B = getBasis(R, logT, basisAlphas, notrunc=args.notrunc)

    RData = {'R': R, 'B': B}

    print("Running DEM inversion")
    tic = time.time()
    DEMCube = invertDEMCube(AIACube / scaleFactor, AIAErrors / scaleFactor, RData, args)
    toc = time.time()
    print("Finished in %.1f seconds" % (toc-tic))

    # if --extendto8 was requested but XRT was not used, pad DEMCube with zeros
    # (if XRT was used, logT was already extended during inversion)
    if args.extendto8 and not args.xrt:
        # DEMCube currently has 18 bins (0-17), need to extend to 26 (0-25)
        logTExpand = np.array([7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0])
        n_extend = len(logTExpand)
        DEMCube = np.vstack([DEMCube, np.zeros((n_extend, DEMCube.shape[1], DEMCube.shape[2]))])
        logT = np.hstack([logT, logTExpand])
        print(f"padded DEMCube with {n_extend} zero bins for extended logT (--extendto8 without XRT)")

    # save the file
    if args.target != "-":
        if not os.path.exists(os.path.dirname(args.target)):
            os.makedirs(os.path.dirname(args.target))
        
        # compression settings
        # DEM: use BitRound to quantize to 12 bits (~3-4 decimals) + bitshuffle compression
        dem_compressor = Blosc(cname='zstd', clevel=5, shuffle=2)
        dem_filters = [BitRound(keepbits=12)]
        
        # AIA: standard compression (no BitRound to preserve precision)
        aia_compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
        
        # compress DEM with BitRound filter
        DEMCubeShape = DEMCube.shape
        DEMCubeRounded = dem_filters[0].encode(DEMCube.astype(np.float32))
        DEMCubeEncode = dem_compressor.encode(DEMCubeRounded)
        
        # for noisy runs (noisy > 0), drop AIA and AIAErrors to save space
        if args.noisy > 0:
            print("noisy run detected: saving DEM only (dropping AIA and AIAErrors)")
            np.savez(args.target,
                DEMCube=DEMCubeEncode, DEMCubeShape=DEMCubeShape,
                logT=logT, scaleFactor=scaleFactor)
        else:
                # save original AIA (before noise), not the noisy version
                if args.decimate != 1:
                    # use the original before decimation
                    AIACubeSave = AIACubeOrig / scaleFactor
                    AIAErrorsSave = AIAErrorsOrig / scaleFactor
                else:
                    # no modifications
                    AIACubeSave = AIACube / scaleFactor
                    AIAErrorsSave = AIAErrors / scaleFactor
                
                AIACubeShape = AIACubeSave.shape
                AIACubeEncode = aia_compressor.encode(AIACubeSave.astype(np.float32))
                AIAErrorsEncode = aia_compressor.encode(AIAErrorsSave.astype(np.float32))
                
                np.savez(args.target, 
                    DEMCube=DEMCubeEncode, DEMCubeShape=DEMCubeShape, 
                    AIACube=AIACubeEncode, AIAErrors=AIAErrorsEncode, AIACubeShape=AIACubeShape,
                    logT=logT, scaleFactor=scaleFactor)
            
            #if args.decimate != 1:
            #    np.savez(args.target, DEMCube=DEMCube, logT=logT, AIACube=AIACubeOrig / scaleFactor, AIAErrors=AIAErrorsOrig / scaleFactor, scaleFactor=scaleFactor)
            #else:
            #    np.savez(args.target, DEMCube=DEMCube, logT=logT, AIACube=AIACube / scaleFactor, AIAErrors=AIAErrors / scaleFactor, scaleFactor=scaleFactor)

    if args.visTarget:
        if not os.path.exists(args.visTarget):
            os.makedirs(args.visTarget)
        setupPage(args.visTarget, logT, False, AIACube.shape[0])

        if args.zerochill:
            print("No relaxation was done, so interpolating missing values")
            DEMCube = nnInterpNaN(DEMCube)

        dumpSynthesis(args.visTarget, "synth", AIACube / scaleFactor, DEMCube, R, wavelengths)

        for ci in range(DEMCube.shape[0]):
            DEMChannel = DEMCube[ci,:,:,]
            plt.imsave(os.path.join(args.visTarget, "%d_vis.png" % ci), DEMChannel**0.5, vmin=0, vmax=np.nanmax(DEMChannel**0.5))

        dumpDiagnostics(args.visTarget, DEMCube, logT)

