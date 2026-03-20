import numpy as np
import os
import sys
import sunpy.map
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from aiapy.calibrate.utils import get_pointing_table, get_correction_table, get_error_table
from aiapy.calibrate import register, update_pointing, degradation, estimate_error
import astropy.units as u
from types import SimpleNamespace
import torch
 
# just a prototype utils file with a bunch of functions

wavelengths = [94, 131, 171, 193, 211, 335]

def processIndAIAData(datePath, args):
    """
    process individual AIA data point
    no deconvolution

    input: 
    datePath: FULL path to the AIA data folder, e.g. /data/input/train/20240520_1200
    args: arguments for the main script, better future use

    returns:
    AIACube: C X H X W
    """
    # min DNs
    _minimumDNs = 0.5
    
    # grab a file in a folder by the nominal wavelength
    getByWL = lambda p, wl: [fn for fn in os.listdir(p) if fn.endswith("%d.image_lev1.fits" % wl)][0]

    print("reading AIA data from %s" % datePath)
    # get all files for now
    AIAFiles = [os.path.join(datePath, getByWL(datePath, wl)) for wl in wavelengths]
    AIAFits = [fits.open(fn) for fn in AIAFiles]

    AIAMaps = [sunpy.map.Map(fn) for fn in AIAFiles]
    print("AIA maps loaded")

    ###########################################################################
    # Get the information needed to properly scale the images. This needs to be
    # applied at the very end -- the noise model is mainly driven by photon
    # noise, and this needs the original counts (not counts per second)
    exposures = [f[1].header["EXPTIME"] for f in AIAFits]
    print("Exposure times loaded")

    if hasattr(args, "corr_table"):
        if os.path.isfile(args.corr_table):
            correction_table = get_correction_table(args.corr_table)  # local read
            print("Using correction table %s" % args.corr_table)
        else:
            print("Correction table %s not found, falling back to JSOC/SSW")
            correction_table = get_correction_table("JSOC")  # falls back to JSOC/SSW
            # cache for future use (astropy QTable)
            correction_table.write(args.corr_table, format='csv', overwrite=True)
            print("Using correction table from JSOC, cached to %s" % args.corr_table)
            
    print("Correction table loaded")
    degradationFactors = []
    for i in range(len(AIAMaps)):
        degradationFactors.append(degradation(wavelengths[i] * u.angstrom, AIAMaps[i].date, correction_table=correction_table))

    print("Exposure times and degradation factors loaded")
    # the total scale factor (to divide by) is the exposure time and the
    # divisive degradataion factor
    scaleFactor = [exposures[i] * degradationFactors[i] for i in range(len(exposures))]
    scaleFactor = np.array(scaleFactor).reshape(-1, 1, 1)

    # Now handle loading the data, updating the pointing, potential
    # deconvolving, and then registering the images
    print("Updating pointing and registering AIA maps")
    pointing_table = get_pointing_table("JSOC", time_range=(AIAMaps[0].date - 12 * u.h, AIAMaps[0].date + 12 * u.h))
    
    print("Applying degradation correction")
    # no deconvolution
    AIAMaps = [register(update_pointing(m, pointing_table=pointing_table)) for m in AIAMaps]

    AIAErrors = []
    # regular photon noise error
    for i in range(len(AIAMaps)):
        AIAErrors.append(np.maximum(AIAMaps[i].data, _minimumDNs)**0.5)
        
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

    if hasattr(args, "crop") and args.crop != "":
        cropSy, cropSx, cropH, cropW = [int(v) for v in args.crop.split(",")]
        AIACube = AIACube[:, cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]
        AIAErrors = AIAErrors[: cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]

    return AIACube / scaleFactor, AIAErrors / scaleFactor, scaleFactor

# visualize an indivvidual AIACube

def plotAIACube(AIACube, title="AIACube", transformed=False):
    # plot the AIACube
    # AIACube: 6x4096x4096
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        cm = plt.get_cmap("sdoaia%d" % wavelengths[i])
        if not transformed:
            ax.imshow(np.maximum(0,AIACube[i,:,:])**0.5,
                        vmin=0, vmax=np.nanmax(AIACube[i,:,:])**0.5, cmap=cm)
        else:
            ax.imshow(np.maximum(0,AIACube[i,:,:]),
                        vmin=0, vmax=np.nanmax(AIACube[i,:,:]), cmap=cm)
        ax.set_title(f"Wavelength: {wavelengths[i]} Å")
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# now let's do the individual for DEMCubes

def processIndDEMData(datePath, args):
    """
    process individual AIACube and DEMCube, which are saved as npz
    """

    onlyDate = os.path.basename(datePath)

    # load the DEMCube
    DEMCube = np.load(os.path.join(datePath, f"{onlyDate}.npz"))
    
    # crop if needed
    if args.crop != "":
        cropSy, cropSx, cropH, cropW = [int(v) for v in args.crop.split(",")]
        DEMCube = DEMCube[:, cropSy:(cropSy+cropH), cropSx:(cropSx+cropW)]


    if args.preprocess == 'sqrt':
        # apply square root
        tf_AIACube = np.maximum(0, DEMCube["AIACube"])**0.5
        tf_DEMCube = np.maximum(0, DEMCube["DEMCube"])**0.5

    elif args.preprocess == 'max':
        # only max
        tf_AIACube = np.maximum(0, DEMCube["AIACube"])

        # make everything positive or 0 (there might be some small negative values really close to 0, this is good for numerical stability)
        tf_DEMCube = np.maximum(0, DEMCube["DEMCube"])
    elif args.preprocess == '':
        # no transformation
        tf_AIACube = DEMCube["AIACube"]
        tf_DEMCube = DEMCube["DEMCube"]

    # return AIACube, DEMCube, logT
    return tf_AIACube, tf_DEMCube, DEMCube["logT"]

def reconstruct_cube(patches, cube_shape, numpy=False):
    """
    Args:
        patches: 4D tensor of shape (num_patches, num_channels, patch_size, patch_size)
        cube_shape: tuple of shape (num_channels, height, width)
    Returns:
        cube: 3D tensor of shape (num_channels, height, width)
    """
    num_patches = patches.shape[0]
    num_channels = patches.shape[1]
    patch_size = patches.shape[2]
    cube = torch.zeros(cube_shape)

    # loop over patches and place them in the cube
    # for i in range(num_patches):
    #     x = i // (cube_shape[1] // patch_size)
    #     y = i % (cube_shape[1] // patch_size)
    #     cube[:, x * patch_size:(x + 1) * patch_size, y * patch_size:(y + 1) * patch_size] = patches[i]

    # or do it with pytorch fold
    # convert patches to tensor of shape [1, C * patch_size * patch_size, num_patches]
    patches_tensor = patches.reshape(num_patches, -1).T.unsqueeze(0)
    
    # fold back to image
    image_tensor = torch.nn.functional.fold(
        patches_tensor, output_size=(cube_shape[1], cube_shape[2]), kernel_size=patch_size, stride=patch_size
    )

    cube = image_tensor.squeeze(0)
    
    if numpy:
        cube = cube.cpu().numpy()
    else:
        cube = cube.to(patches.device)

    return cube

def reconstruct_all_cubes(patches, num_patches_per_image, cube_shape):
    """
    Args:
        patches: 4D tensor of shape (num_patches, num_channels, patch_size, patch_size)
        num_patches_per_image: number of patches per image
        cube_shape: tuple of shape (num_channels, height, width)
    Returns:
        cubes: list of 3D tensors of shape (num_channels, height, width)
    """
    if patches.shape[0] % num_patches_per_image != 0:
        raise ValueError("Number of patches must be divisible by num_patches_per_image")
    cubes = []
    for i in range(len(patches) // num_patches_per_image):
        start = i * num_patches_per_image
        end = (i + 1) * num_patches_per_image
        cube = reconstruct_cube(patches[start:end], cube_shape)
        cubes.append(cube)
    return cubes

def metrics_test(dem, dem_pred):
    dem = dem.reshape(dem.shape[0], -1)
    dem_pred = dem_pred.reshape(dem_pred.shape[0], -1)
    mask = np.isnan(dem) | np.isnan(dem_pred)
    dem = dem[~mask]
    dem_pred = dem_pred[~mask]
    mae = np.mean(np.abs(dem - dem_pred))
    mse = np.mean((dem - dem_pred) ** 2)
    log_dem = np.log10(np.maximum(dem, 1e-8))
    log_pred = np.log10(np.maximum(dem_pred, 1e-8))
    log_mae = np.mean(np.abs(log_dem - log_pred))
    log_mse = np.mean((log_dem - log_pred) ** 2)
    total_em_diff = np.mean(np.abs(np.sum(dem, axis=0) - np.sum(dem_pred, axis=0)))
    return {
        'mae': mae,
        'mse': mse,
        'log_mae': log_mae,
        'log_mse': log_mse,
        'total_em_diff': total_em_diff,
    }

def dumpDEMJointPDF(target, dem_truth, dem_pred, transformed=None):
    """
    Dump the joint PDF of the DEM predicted vs ground truth
    transformed: either name of transformation, or array of strings(e.g. ["sqrt", None])
    """

    if not os.path.exists(target):
        os.makedirs(target)

    # untransform the DEM cubes if necessary
    if transformed is not None:
        if transformed[0] == 'sqrt':
            dem_cube_truth = dem_cube_truth**2
        if transformed[1] == 'sqrt':
            dem_cube = dem_cube**2
            
    for i in range(dem_truth.shape[0]):
        gt = dem_truth[i].ravel()
        pd = dem_pred[i].ravel()

        # avoid zeros/nans before counting
        gt = np.maximum(gt, 1e-8)
        pd = np.maximum(pd, 1e-8)

        # mask 
        mask = (gt > 1) & (pd > 1) & ~np.isnan(gt) & ~np.isnan(pd)

        x = np.log10(gt[mask])
        y = np.log10(pd[mask])

        # R2
        r2 = np.corrcoef(x, y)[0,1]**2

        # count zeros: ground ≤1 region
        zero_mask = gt <= 1
        tn = np.sum(zero_mask & (pd <= 1))
        fp = np.sum(zero_mask & (pd > 1))
        fn = np.sum((gt > 1) & (pd <= 1))

        # compute rates
        true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        false_negative_rate = fn / (fn + tn) if (fn + tn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        plt.figure(figsize=(4,4))
        plt.hexbin(gt[mask], pd[mask], gridsize=200, bins='log', xscale='log', yscale='log', cmap='turbo', extent=[0, 2, 0, 2])
        plt.plot([1, 100], [1, 100], 'k--')

        # put and R^2 in the title
        plt.title(r"$R^2 = %.2f$" % r2)
        plt.xlabel(r"$DEM_{lp}$")
        plt.ylabel(r"$DEM_{model}$")
        plt.axis('square')

        txt = f"tn={true_negative_rate*100:.2f}, fp={false_positive_rate*100:.2f}\nfn={false_negative_rate*100:.2f}"

        plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
                 va='top', ha='left', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray"))

        plt.show()

def getBasis(R, logT, alphas=[0.0, 0.1, 0.2, 0.6]):
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
            basisResult[basisResult<0.04] = 0
            basis[:,col] = basisResult

    return basis

def compute_photon_error_bounds(aia_obs, aia_errors, tolfac=1.4, scale=1):
    """
    Compute the photon error bounds for AIA data as 
    tolfac: how much to multiply the error by to produce constraints
    scale: scale factor for bounds, default is 1 (no scaling)
    Returns:
        lb: lower bound of the error
        ub: upper bound of the error
    """
    # flatten the AIA observations and errors
    # permute to (B, H, W, 6) then flatten spatial dimensions
    aia_obs_flat = aia_obs.permute(0, 2, 3, 1).reshape(aia_obs.size(0), -1, 6)
    aia_errors_flat = aia_errors.permute(0, 2, 3, 1).reshape(aia_errors.size(0), -1, 6)
    
    tol = tolfac * aia_errors_flat
    lb = aia_obs_flat - scale * tol
    ub = aia_obs_flat + scale * tol

    # return lower and upper bounds
    return lb, ub

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

    plt.imsave(os.path.join(target, "mean_logt.png"), meanLogT[0,:,:], vmin=np.min(logT), vmax=np.max(logT), cmap='inferno')
    plt.imsave(os.path.join(target, "std_logt.png"), stdLogT[0,:,:], vmin=0, vmax=0.5*logTRange, cmap='inferno')

def compute_synthesis(DEMCube, R):
    '''
    Compute the synthesis of the DEM cube using the basis R.
    '''
    resynth = torch.tensordot(R, DEMCube, dims=([1], [0]))  # shape: (C, H, W)
    return resynth

def unfold_tensor(array, patch_size=256, stride=256):
    tensor = torch.from_numpy(array.copy()).unsqueeze(0)  # [1, C, H, W]
    patches = torch.nn.functional.unfold(
        tensor, kernel_size=patch_size, stride=stride
    )  # [1, C*ps*ps, N]
    patches = patches.squeeze(0).T
    patches = patches.reshape(-1, array.shape[0], patch_size, patch_size)
    return patches.numpy()

def create_dem_bins(vmin=0, vmax=100, n_bins=64, spacing='linear'):
    """Create DEM value bins"""
    if spacing == 'linear':
        bins = np.linspace(vmin, vmax, n_bins + 1)
    elif spacing == 'sqrt':
        edges = (np.linspace(np.sqrt(vmin), np.sqrt(vmax), n_bins + 1)**2)
        edges[0]  = 0.0
        edges[-1] = vmax
        bins = edges
    elif spacing == 'log':
        eps = 1.0  # this is default, lets see!
        edges = np.exp(np.linspace(np.log(eps), np.log(vmax + eps), n_bins + 1)) - eps
        edges[0] = 0.0
        edges[-1] = vmax
        bins = edges
    else:
        raise ValueError(f"Unknown spacing: {spacing}")
    
    return bins

def dem_values_to_bins(dem_values, bins):
    """Convert DEM values to bin indices"""

    # check data type
    if not torch.is_tensor(dem_values):
        dem_values = torch.tensor(dem_values, dtype=torch.float32)
    if not torch.is_tensor(bins):
        bins = torch.tensor(bins, dtype=torch.float32)
    
    # clip values to bins range
    dem_clipped = torch.clamp(dem_values, bins[0], bins[-1])

    # get bin indices
    bin_indices = torch.searchsorted(bins, dem_clipped, right=False) - 1

    # ensure indices are within valid range
    bin_indices = torch.clamp(bin_indices, 0, len(bins) - 2)

    return bin_indices

def bins_to_dem_values(bin_indices, bins):
    """Convert bin indices back to DEM values (bin centers)"""
    """this won't give exactly the original values, but the center of the bin"""
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers[bin_indices]

@torch.no_grad()
def quantiles_from_pmf(logits, bins, q_low=0.05, q_med=0.50, q_high=0.95, use_bin_edges=False):
    """
    compute quantiles from classification logits via linear interpolation within bins
    
    args:
        logits: (B, K, H, W) or (K, H, W) - classification logits OR probabilities
                if sum to 1 along dim 1, treated as pmf; otherwise softmaxed
        bins: (K+1,) bin edges (monotone increasing) - as returned by create_dem_bins
        q_low: lower quantile (default 0.05 for 5th percentile)
        q_med: median quantile (default 0.50)
        q_high: upper quantile (default 0.95 for 95th percentile)
        use_bin_edges: if True, use full bin extent [lower, upper] as CI for peaked distributions
    
    returns:
        low, med, high: each of shape (B, H, W) or (H, W) depending on input
    """
    # handle both 3d and 4d inputs
    is_4d = logits.ndim == 4
    if not is_4d:
        logits = logits.unsqueeze(0)  # add batch dimension
    
    B, K, H, W = logits.shape
    
    # ensure bins is a tensor on same device
    if not torch.is_tensor(bins):
        bins = torch.tensor(bins, dtype=logits.dtype, device=logits.device)
    else:
        bins = bins.to(device=logits.device, dtype=logits.dtype)
    
    # validate shapes
    assert bins.shape[0] == K + 1, f"bins must have K+1={K+1} edges, got {bins.shape[0]}"
    assert bins.ndim == 1, f"bins must be 1D, got shape {bins.shape}"
    
    # check if already normalized (pmf) or needs softmax (logits)
    # consider it a pmf if sum is close to 1 (within 1e-3 tolerance)
    sum_along_k = logits.sum(dim=1)  # (B, H, W)
    is_already_pmf = torch.allclose(sum_along_k, torch.ones_like(sum_along_k), atol=1e-3)
    
    if is_already_pmf:
        pmf = logits
    else:
        pmf = torch.softmax(logits, dim=1)  # (B, K, H, W)
    
    # compute cdf along bins
    cdf = pmf.cumsum(dim=1)  # (B, K, H, W)
    
    def q_from_cdf(cdf, q):
        # find first index where cdf >= q
        idx = (cdf < q).sum(dim=1)  # (B, H, W), values in [0..K]
        idx = idx.clamp(0, K - 1)    # cap at last bin
        
        # gather F_k
        idx_expanded = idx.unsqueeze(1)  # (B, 1, H, W)
        Fk = cdf.gather(1, idx_expanded).squeeze(1)  # (B, H, W)
        
        # gather F_{k-1} with explicit handling: F_{-1} = 0
        in_first_bin = (idx == 0)
        idx_prev = (idx - 1).clamp(min=0)  # safe indexing
        idx_prev_expanded = idx_prev.unsqueeze(1)
        Fkm1 = cdf.gather(1, idx_prev_expanded).squeeze(1)  # (B, H, W)
        Fkm1 = torch.where(in_first_bin, torch.zeros_like(Fkm1), Fkm1)  # F_{-1} = 0
        
        # bin edges for interpolation
        bin_lower = bins[:-1]  # (K,)
        bin_upper = bins[1:]   # (K,)
        lower_k = bin_lower[idx]  # (B, H, W)
        upper_k = bin_upper[idx]  # (B, H, W)
        
        # linear interpolation: alpha = (q - Fkm1) / (Fk - Fkm1)
        # handle zero-mass bins: if Fk == Fkm1, set alpha=0 (use lower edge)
        denom = Fk - Fkm1
        zero_mass = denom < 1e-12
        alpha = torch.where(zero_mass, torch.zeros_like(denom), (q - Fkm1) / denom)
        alpha = alpha.clamp(0, 1)
        
        # interpolate within bin
        qval = lower_k + alpha * (upper_k - lower_k)
        
        return qval
    
    low = q_from_cdf(cdf, q_low)
    med = q_from_cdf(cdf, q_med)
    high = q_from_cdf(cdf, q_high)
    
    # optional: special handling for bin 0 only
    if use_bin_edges:
        # find the most likely bin (argmax of pmf)
        most_likely_bin = torch.argmax(pmf, dim=1)  # (B, H, W)
        
        # for pixels where >20% mass is in one bin
        max_prob = torch.max(pmf, dim=1)[0]  # (B, H, W)
        peaked = max_prob > 0.2
        
        # special handling for bin 0 only: set lower = 0, keep computed upper
        is_bin_0 = (most_likely_bin == 0) & peaked
        
        if is_bin_0.any():
            # for bin 0: lower = 0, upper = original q_from_cdf (keep computed quantile)
            # for all other bins: use computed quantiles as-is
            low = torch.where(is_bin_0, torch.zeros_like(low), low)
    
    # remove batch dimension if input was 3d
    if not is_4d:
        low = low.squeeze(0)
        med = med.squeeze(0)
        high = high.squeeze(0)
    
    return low, med, high

@torch.no_grad()
def ci_coverage(y, lo, hi):
    """
    compute coverage rate of confidence interval
    
    args:
        y: ground truth values (any shape)
        lo: lower bound of confidence interval (same shape as y)
        hi: upper bound of confidence interval (same shape as y)
    
    returns:
        rate: scalar coverage rate (fraction of points inside interval)
        inside: boolean mask of same shape as y (useful for visualization)
    """
    inside = (y >= lo) & (y <= hi)
    rate = inside.float().mean()
    return rate, inside

@torch.no_grad()
def relative_error(y, pred, eps=1e-12):
    """
    compute relative error: |y - pred| / max(|y|, eps)
    
    args:
        y: ground truth values (any shape)
        pred: predicted values (same shape as y)
        eps: minimum denominator to avoid division by zero
    
    returns:
        rel_err: relative error map (same shape as y)
        mean_rel_err: scalar mean relative error
    """
    denom = y.abs().clamp_min(eps)
    rel_err = (y - pred).abs() / denom
    return rel_err, rel_err.mean()

@torch.no_grad()
def relative_uncertainty(lo, hi, ref, eps=1e-12):
    """
    compute relative uncertainty: (hi - lo) / |ref|
    
    args:
        lo: lower bound of confidence interval (any shape)
        hi: upper bound of confidence interval (same shape as lo)
        ref: reference magnitude (same shape, typically y or pred)
        eps: minimum denominator to avoid division by zero
    
    returns:
        rel_unc: relative uncertainty map (same shape as inputs)
        mean_rel_unc: scalar mean relative uncertainty
    """
    width = (hi - lo).clamp_min(0)
    denom = ref.abs().clamp_min(eps)
    rel_unc = width / denom
    return rel_unc, rel_unc.mean()

@torch.no_grad()
def signed_distance_to_ci(y, lo, hi):
    """
    compute signed distance from y to confidence interval [lo, hi]
    positive if above interval, negative if below, zero if inside
    
    args:
        y: ground truth values (any shape)
        lo: lower bound of confidence interval (same shape as y)
        hi: upper bound of confidence interval (same shape as y)
    
    returns:
        signed_dist: signed distance map (same shape as y)
                     >0 if y > hi (above interval)
                     =0 if lo <= y <= hi (inside interval)
                     <0 if y < lo (below interval)
                     useful for visualization with diverging colormap (bwr)
    """
    above = (y - hi).clamp_min(0)  # >0 only if y > hi
    below = (lo - y).clamp_min(0)  # >0 only if y < lo
    
    # compute signed distance: positive if above, negative if below, zero if inside
    signed_dist = above - below
    return signed_dist