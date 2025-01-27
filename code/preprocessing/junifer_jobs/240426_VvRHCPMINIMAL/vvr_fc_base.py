"""Provide base class for voxel- and ROI-wise denoising/confound removal."""

# Authors: Tobias Muganga <t.muganga@fz-juelich.de>
# License: AGPL

# imports
import os
import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Union
from nilearn.image import math_img
from nilearn.signal import clean
from nilearn.maskers import NiftiMasker
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import EmpiricalCovariance
from datetime import datetime
from scipy.stats import zscore

from junifer.api.decorators import register_marker
from junifer.utils import logger
from junifer.data import get_parcellation
from junifer.markers.base import BaseMarker


@register_marker
class VvRConnectivityBase(BaseMarker):
    """Base class for voxel- and region-wise denoising.
    Implements all important functions from BOLD TS extraction to
    calculation of functional connectivity.

    Inheriting class only need to call methods in the sequence the steps
    should be performed in for their compute function.

    Parameters:
    -----------
    parcellation : Union[str, List[str]]
        Parcellation name or list of parcellation names.

    detrend : bool, optional
        Whether to detrend the data during cleaning step.
        Defaults to True.

    standardize : bool, optional
        Whether to standardize the data during cleaning step.
        Defaults to False.

    aggr_method : str, optional
        Method for aggregating voxel signals within a parcel.
        Accepts "mean" and "eigenvar". Defaults to "mean".

    aggr_demean : bool, optional
        Whether to demean voxel signals before aggregation into parcel signals.
        Only considered for experimental reasons in "eigenvar" case.
        Defaults to False.

    cor_method : str, optional
        Method for computing correlation. Defaults to "correlation".

    masks : Union[str, Dict, List[Union[Dict, str]], None], optional
        Masks to be applied to the data. Defaults to None.

    low_pass : Optional[float], optional
        Low pass frequency for filtering during cleaning step.
        Defaults to None.

    high_pass : Optional[float], optional
        High pass frequency for filtering during cleaning step.
        Defaults to None.

    t_r : Optional[float], optional
        Repetition time. If not specified read the TR from header
        of nifti file. Defaults to None.

    cor_method_params : Optional[Dict], optional
        Additional parameters for the correlation method.
        If set to 'empirical' FC will be calculated using Maximum likelihood
        covariance estimator. Defaults to None.

    name : Optional[str], optional
        Name of the marker. Defaults to None.

    """

    def __init__(
        self,
        parcellation: Union[str, List[str]],
        detrend: bool = True,
        standardize: bool = False,
        aggr_method: str = "mean",
        aggr_demean: bool = False,
        cor_method: str = "correlation",
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None,
        t_r: Optional[float] = None,
        cor_method_params: Optional[Dict] = None,
        name: Optional[str] = None,
    ) -> None:
        self._detrend = detrend
        self._standardize = standardize
        self._low_pass = low_pass
        self._high_pass = high_pass
        self._t_r = t_r
        self.masks = masks
        if not isinstance(parcellation, list):
            parcellation = [parcellation]
        self.parcellation = parcellation
        self.aggr_method = aggr_method
        self.aggr_demean = aggr_demean
        self.cor_method = cor_method
        self.cor_method_params = cor_method_params or {}
        super().__init__(name=name, on="BOLD")

    def get_valid_inputs(self) -> List[str]:
        """Get valid data types for input.

        Returns
        -------
        list of str
            The list of data types that can be used as input for this marker.

        """
        return ["BOLD"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type.

        Parameters
        ----------
        input_type : str
            The data type input to the marker.

        Returns
        -------
        str
            The storage type output by the marker.

        """
        return "matrix"

    def _pick_confounds(self, input: Dict[str, Any]):
        """Picks pre-computed confounds according to the dataset used from
        files stored in '/data/group/appliedml/data/HCP1200_Confounds_tsv/'
        directory on compute cluster (Juseless).

        This method extracts confounds based on the input data, which
        includes subject, session, phase encoding, and file path.
        The confounds are selected based on whether the file name indicates
        it's from the HCP's ICA-FIX preprocessed or the minimally preprocessed
        dataset.

        Confounds are loaded from a predefined directory and NaN values
        are replaced with zeros.

        Parameters
        ----------
        input: dict
            Junifers data object containing
            metadata (subject, task, phase encoding) and file path for
            identification of dataset and confound file.

        Returns
        -------
        confounds: numpy.ndarry
            Selected confounds.

        Raises
        ------
        AssertionError
            If the directory containing confounds does not exist or if
            the confounds file does not exist.

        """

        assert "meta" in input and isinstance(
            input["meta"], dict
        ), "Input should contain metadata."
        assert (
            "subject" in input["meta"]["element"]
        ), "Subject information is missing in metadata."
        assert (
            "task" in input["meta"]["element"]
        ), "Session information is missing in metadata."
        assert (
            "phase_encoding" in input["meta"]["element"]
        ), "Phase encoding information is missing in metadata."
        assert "path" in input, "Input should contain a file path."

        # Select meta data from junifer object
        subject = input["meta"]["element"]["subject"]
        session = input["meta"]["element"]["task"]
        phase_encoding = input["meta"]["element"]["phase_encoding"]
        file = str(input["path"])

        # Ensure the file name format is as expected
        assert file.endswith(
            ".nii.gz"
        ), "Input file should be in gunzip NIfTI format."

        logger.debug(
            f"Using {subject}, {session}, {phase_encoding}"
            f" to grab confounds for file {file}"
        )

        # Specify confound file and regressors according to dataset used
        confounds_names = ["WM", "CSF", "GS"]

        if file.split("_")[-1] == "clean.nii.gz":
            logger.info("Getting confounds for FIX dataset.")
            confound_file = os.path.join(
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"rfMRI_{session}_{phase_encoding}",
                f"Confounds_{subject}.tsv",
            )

        else:
            logger.info("Getting confounds for MINIMAL dataset.")
            confound_file = os.path.join(
                f"{subject}",
                "MNINonLinear",
                "Results",
                f"rfMRI_{session}_{phase_encoding}",
                f"Confounds_{subject}_noFIX.tsv",
            )
            confounds_names.append("RP")

        logger.debug(f"Getting confounds file: {confound_file}")

        # Specify confound directory on Juseless
        aml_dir = os.path.join(
            "/data", "group", "appliedml", "data", "HCP1200_Confounds_tsv"
        )
        assert os.path.isdir(aml_dir), "Make sure you provided the right path."

        confound_path = os.path.join(aml_dir, confound_file)
        assert os.path.isfile(confound_path), "Confounds file does not exist."

        # Reading confounds from file (replace NaN with zero)
        logger.info(f"Loading confound types: {confounds_names}")
        confounds = np.nan_to_num(
            get_aml_confounds(confounds_names, confound_path)
        )

        return confounds

    def _extract_timeseries(
        self,
        input: Dict[str, Any],
        extra_input: Optional[Dict[str, Any]] = None,
    ):
        """Extracts 2D time series data from the input nifti image.

        Parameters
        ----------
        input : dict
            Dictionary containing the ``BOLD`` value from the
            Junifer Data object.
        extra_input : dict, optional
            Dictionary containing the rest of the Junifer Data object.

        Returns
        -------
        bold_ts : numpy.array
            Voxel-wise bold time series with shape time points X voxels.

        """

        # Get bold image data (nifti format) from junifer object
        bold_img = input["data"]

        # If repetition time is not specified, extract from nifti header
        if self._t_r is None:
            logger.info("No `t_r` specified, using t_r from nifti header")
            zooms = bold_img.header.get_zooms()  # type: ignore
            self._t_r = zooms[3]
            logger.info(f"Read t_r from nifti header: {self._t_r}")

        # Get parcellation tailored to target image
        self._parcellation_img, self._labels = get_parcellation(
            parcellation=self.parcellation,
            target_data=input,
            extra_input=extra_input,
        )

        # Get binarized parcellation image for masking
        self._parcellation_bin = math_img(
            "img != 0", img=self._parcellation_img
        )

        # Apply masking and extract time series
        self._masker = NiftiMasker(
            mask_img=self._parcellation_bin,
            target_affine=bold_img.affine,
        )
        bold_ts = self._masker.fit_transform(bold_img)

        logger.debug(f"Extracted time series with dimensions: {bold_ts.shape}")

        return bold_ts

    def _clean(self, timeseries, input):
        """Clean 2D time series applying the following order:
            1. Detrend
            2. Temporal filtering
            3. Linear confound regression
            4. Standardize

        Parameters
        ----------
        timeseries : numpy.ndarray
            Time series data as 2D numpy.array for cleaning.
        input : dict
            Dictionary containing the Junifer Data object. Must include the
            ``BOLD_confounds`` key. Will be passed to pick_confounds for
            confound selection.

        Returns
        -------
        cleaned bold time sereis : numpy.array
            denoised bold time series according to settings.

        """

        # Selecting confounds to remove
        confounds = self._pick_confounds(input)

        assert (
            confounds.shape[0] == timeseries.shape[0]
        ), "Mismatch in timeseries and confound dimensions"
        logger.info("Cleaning time series with settings")
        logger.info(f"\tdetrend: {self._detrend}")
        logger.info(f"\tlow_pass: {self._low_pass}")
        logger.info(f"\thigh_pass: {self._high_pass}")
        logger.info(f"\t{confounds.shape[1]} regressors")
        logger.info(f"\tstandardize: {self._standardize}")

        logger.debug(f"TS before cleaning: {timeseries[0,0:10]}")

        clean_ts = clean(
            signals=timeseries,
            detrend=self._detrend,
            standardize=self._standardize,
            confounds=confounds,
            low_pass=self._low_pass,
            high_pass=self._high_pass,
            t_r=self._t_r,
        )

        logger.debug(f"TS after cleaning: {clean_ts[0,0:10]}")

        return clean_ts

    def _aggregate(
        self,
        timeseries,
        input: Dict[str, Any],
        extra_input: Optional[Dict] = None,
    ):
        """Aggregates the provided time series data based on the
        specified method.

        Parameters
        ----------
        timeseries : numpy.array
            Time series data as 2D numpy.array for aggregation using method
            specified in yaml file.
        input : dict
            Dictionary containing the ``BOLD`` value from the
            Junifer Data object.
        extra_input : dict, optional
            Dictionary containing the rest of the Junifer Data object.


        Returns
        -------
        aggr_ts : numpy.array
            Aggregated time series data.

        """

        # Ensure a valid aggregation method is specified
        assert self.aggr_method in [
            "eigenvar",
            "mean",
        ], "Specify a valid aggregation method: 'eigenvar' or 'mean'"

        # Check if time series has expected shape (tpoints x numvoxel)
        assert (
            timeseries.shape[0] == 1200
        ), "Timeseries has unexpected format or length."

        # Ensure that needed parcellation image and masker exist
        assert self._parcellation_img is not None, "Parcellation image needed."
        assert self._masker is not None, "Masker object needed."

        # # Format the extracted labels to readable format
        # split = [x.split("~") for x in self._labels]
        # rejoined = ["_".join([str(int(float(x))) for x in y]) for y in split]
        # self._labels = rejoined

        # Mask the parcellation
        parcellation_values = np.squeeze(
            self._masker.fit_transform(self._parcellation_img)
        ).astype(int)

        # Get the values for each parcel and apply agg function
        logger.info(f"Computing {self.aggr_method} ROI time series")
        logger.info(f"Demean set to {self.aggr_demean}")
        out_values = []

        # Iterate over the parcels (existing)
        for t_v in range(1, len(self._labels) + 1):
            idx = parcellation_values == t_v
            ts = timeseries[:, idx.flatten()]

            # replacing nan values with 0
            if np.isnan(ts).sum() > 0:
                print(
                    f"{np.isnan(ts).sum()} NaN values detected for "
                    f"region {t_v}. Replacing with 0!"
                )
                ts = ts.fillna(0)

            if self.aggr_method == "eigenvar":
                # Aggregation into 1st eigenvariate time series
                aggr_ts = get_first_eigen_ts(ts, self.aggr_demean)

            elif self.aggr_method == "mean":
                # Aggregation into average time series
                aggr_ts = np.mean(ts, axis=1)

            out_values.append(aggr_ts)

        # Transpose to match the needed shape (tpoints X numvoxels)
        aggr_ts = np.array(out_values).T

        logger.debug(f"Shape of aggregated TS: {aggr_ts.shape}")

        return aggr_ts

    def connectivity(self, timeseries):
        """Compute functional connectivity
        from provided time series data.

        Parameters
        ----------
        timeseries : numpy.array
            2D array of shape (timepoints, regions)
            containing the time series data for calculating
            functional connectivity.

        Returns
        -------
        conn : numpy.array
            Functional connectivity matrix.

        """

        logger.info("Calculating connectivity!")

        # Compute correlation
        # If specified, use Maximum likelihood covariance estimator
        if self.cor_method_params["empirical"]:
            connectivity = ConnectivityMeasure(
                cov_estimator=EmpiricalCovariance(),  # type: ignore
                kind=self.cor_method,
                standardize=False,
            )
        else:
            # Else, use default LedoitWolf estimator
            connectivity = ConnectivityMeasure(kind=self.cor_method)

        # replacing nan values with 0
        logger.debug(
            f"Timeseries has zeros in {len(np.argwhere(timeseries == 0))}"
            f" places and {len(np.unique(np.argwhere(timeseries == 0)[:,1]))}"
            " different regions BEFORE zscoreing!"
        )
        
        # Manually normalize ts before connectivity calculation to correct for
        # nilearn behavior (uses population mean instead of sample mean)
        timeseries = zscore(timeseries, axis=0)

        # replacing nan values with 0
        if np.isnan(timeseries).sum() > 0:
            logger.debug(
                f"{np.isnan(timeseries).sum()} NaN values detected AFTER"
                " normalization. Replacing with 0!"
            )
            timeseries = np.nan_to_num(timeseries)

        conn = connectivity.fit_transform([timeseries])[0]

        # replacing nan values with 0
        if np.isnan(conn).sum() > 0:
            logger.info(
                f"{np.isnan(conn).sum()} NaN values detected in FC."
                " Replacing with 0!"
            )
            conn = np.nan_to_num(conn)

        return conn


def get_first_eigen_ts(parcel_data, demean=False):
    """Extract 1st eigenvariate time series from provided voxel time series.

    This code is directly adapted from:
    https://github.com/weidongcai/GeneralScripts/blob/0a0253346acffc21849c1f66f4fb2bf7ac79e8de/Python3.5/fMRIAnalysisModule/ROIAnalysisModule.py

    Parameters
    ----------
    parcel_data : tpoints x nvoxels matrix
        tpoints is number of time points, nvoxel is number of voxels
    demean : bool
        whether to mean center the data before decomposition.

    Returns
    -------
    eigents : numpy.array
        tpoints x 1 vector of 1st eigenvariates

    """

    # Ensure that X is a numpy array‚
    voxelts = np.array(parcel_data)

    # Mean center time series, if specified
    if demean:
        voxelts = voxelts - np.mean(voxelts, axis=0)

    # Get the dimensions of the input matrix X
    tpoints, nvoxels = voxelts.shape

    # Check if the number of time points is greater than the number of voxels
    if tpoints > nvoxels:
        # Perform Singular Value Decomposition (SVD) on
        # voxel time series transposed * voxel time series
        u_, s, v = np.linalg.svd(np.dot(voxelts.T, voxelts))
        v = v.T  # Transpose v to match MATLAB and scipy SVD conventions
        v = v[:, 0]  # Extract the first column of v
        u = np.divide(np.dot(voxelts, v), np.sqrt(s[0]))  # Compute u

    else:
        # Perform SVD on voxel time series * voxel time series transposed
        u, s, v_ = np.linalg.svd(np.dot(voxelts, voxelts.T))
        u = u[:, 0]  # Extract the first column of u
        v = np.dot(voxelts.T, np.divide(u, np.sqrt(s[0])))  # Compute v

    # Adjust signs to ensure consistency
    d = np.sign(np.sum(v))
    u = np.dot(u, d)
    v = np.dot(v, d)

    # Compute the final eigenvariate time series Y
    eigents = np.dot(u, np.sqrt(np.divide(s[0], nvoxels)))

    return eigents


def get_aml_confounds(conf_list, confounds_file):
    """Extracts confound regressors from the specified file according to
    the provided list.

    For confounds not of the class realignment parameters (RP) additional
    temporal derivatives and their quare terms are computed and added to
    the list of confounds in alignment with Satterthwait et al., 2013.

    Parameters
    ----------
    conf_list: list(str)
        A list of strings contianing all keywords.
    conf_file: str
        A .tsv filename that we can use to read all confounds.

    Returns
    -------
    confounds: numpy.ndarray
        An array of chosen confounds.

    """

    confounds_df = pd.read_csv(confounds_file, sep="\t")
    confounds_idx = []

    for element in conf_list:
        # Compile a list containing the relevant confounds
        # Exclude columns containing PCA, 50, 80 or 95
        matching = list(
            filter(
                lambda s: "PCA" not in s
                and all(exclude not in s for exclude in ["50", "80", "95"])
                and element in s,
                confounds_df.columns.values,
            )
        )
        confounds_idx.extend(matching)

        # For all non motion params compute derivatives and their squares
        if element != "RP":
            derivative = np.append(0, np.diff(confounds_df[element]))
            norm_deriv = zscore(derivative, axis=0)

            deriv_column_name = f"D{element}"
            confounds_idx.append(deriv_column_name)
            confounds_df[deriv_column_name] = norm_deriv

            squares = np.square(norm_deriv)
            norm_squares = zscore(squares, axis=0)

            sqrt_column_name = f"D{element}^2"
            confounds_idx.append(sqrt_column_name)
            confounds_df[sqrt_column_name] = norm_squares

    # Get unique confounds indices and return confounds.
    confounds_idx = np.unique(confounds_idx)
    confounds_df = confounds_df[confounds_idx]

    print(f"Confound regressors selected: {confounds_df.columns.values}")

    confounds = np.array(confounds_df)

    return confounds


@register_marker
class VvRFCCleanAggr(VvRConnectivityBase):
    """A class for generating functional connectivity (FC) that applies
    denoising on the voxle-wise time series and aggregates the signal after.
    FC is computed from aggregated time series.

    Parameters:
    -----------
    parcellation : Union[str, List[str]]
        Parcellation name or list of parcellation names.

    detrend : bool, optional
        Whether to detrend the data during cleaning step.
        Defaults to True.

    standardize : bool, optional
        Whether to standardize the data during cleaning step.
        Defaults to False.

    aggr_method : str, optional
        Method for aggregating voxel signals within a parcel.
        Accepts "mean" and "eigenvar". Defaults to "mean".

    aggr_demean : bool, optional
        Whether to demean voxel signals before aggregation intoparcel signals.
        Only considered for experimental reasons in "eigenvar" case.
        Defaults to False.

    cor_method : str, optional
        Method for computing correlation. Defaults to "correlation".

    masks : Union[str, Dict, List[Union[Dict, str]], None], optional
        Masks to be applied to the data. Defaults to None.

    low_pass : Optional[float], optional
        Low pass frequency for filtering during cleaning step.
        Defaults to None.

    high_pass : Optional[float], optional
        High pass frequency for filtering during cleaning step.
        Defaults to None.

    t_r : Optional[float], optional
        Repetition time. If not specified read the TR from header
        of nifti file. Defaults to None.

    cor_method_params : Optional[Dict], optional
        Additional parameters for the correlation method.
        If set to 'empirical' FC will be calculated using Maximum likelihood
        covariance estimator. Defaults to None.

    name : Optional[str], optional
        Name of the marker. Defaults to None.

    """

    def __init__(
        self,
        parcellation: Union[str, List[str]],
        detrend: bool = True,
        standardize: bool = True,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None,
        t_r: Optional[float] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
        aggr_method: str = "mean",
        aggr_demean: bool = False,
        cor_method: str = "correlation",
        cor_method_params: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            detrend=detrend,
            standardize=standardize,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            masks=masks,
            parcellation=parcellation,
            aggr_method=aggr_method,
            aggr_demean=aggr_demean,
            cor_method=cor_method,
            cor_method_params=cor_method_params,
            name=name,
        )

    def compute(self, input, extra_input):
        """Perform voxel-wise denoising and compute connectivity.
        1. Time series extraction
        2. Cleaning the voxel-wise signal
        3. Aggregation into region-wise time series
        4. FC generation

        Parameters
        ----------
        input: The input data.

        extra_input: Any additional input data.

        Returns
        -------
            Dict: A dictionary containing the computed connectivity data
            along with other metadata.

        """

        # Create dictionary for output
        out = {}

        # Start a timer
        t_start = datetime.now()

        # Perform voxel-level denoising
        voxelts = self._extract_timeseries(input, extra_input)
        cleants = self._clean(voxelts, input)
        aggregatets = self._aggregate(cleants, input, extra_input)

        # Save FC and processing time
        out["data"] = self.connectivity(aggregatets)

        proc_time = datetime.now() - t_start
        logger.info(f"Time from extraction to connectivity: {proc_time}")

        # Create column names
        out["row_names"] = self._labels
        out["col_names"] = self._labels
        out["matrix_kind"] = "tril"
        out["diagonal"] = False

        return out


@register_marker
class VvRFCAggrClean(VvRConnectivityBase):
    """A class for generating functional connectivity (FC) that applies
    denoising on the region-wise time series. First aggregation is performed
    followed by denoising.
    FC is computed from the cleaned time series.

    Parameters:
    -----------
    parcellation : Union[str, List[str]]
        Parcellation name or list of parcellation names.

    detrend : bool, optional
        Whether to detrend the data during cleaning step.
        Defaults to True.

    standardize : bool, optional
        Whether to standardize the data during cleaning step.
        Defaults to False.

    aggr_method : str, optional
        Method for aggregating voxel signals within a parcel.
        Accepts "mean" and "eigenvar". Defaults to "mean".

    aggr_demean : bool, optional
        Whether to demean voxel signals before aggregation intoparcel signals.
        Only considered for experimental reasons in "eigenvar" case.
        Defaults to False.

    cor_method : str, optional
        Method for computing correlation. Defaults to "correlation".

    masks : Union[str, Dict, List[Union[Dict, str]], None], optional
        Masks to be applied to the data. Defaults to None.

    low_pass : Optional[float], optional
        Low pass frequency for filtering during cleaning step.
        Defaults to None.

    high_pass : Optional[float], optional
        High pass frequency for filtering during cleaning step.
        Defaults to None.

    t_r : Optional[float], optional
        Repetition time. If not specified read the TR from header
        of nifti file. Defaults to None.

    cor_method_params : Optional[Dict], optional
        Additional parameters for the correlation method.
        If set to 'empirical' FC will be calculated using Maximum likelihood
        covariance estimator. Defaults to None.

    name : Optional[str], optional
        Name of the marker. Defaults to None.

    """

    def __init__(
        self,
        parcellation: Union[str, List[str]],
        detrend: bool = True,
        standardize: bool = True,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None,
        t_r: Optional[float] = None,
        masks: Union[str, Dict, List[Union[Dict, str]], None] = None,
        aggr_method: str = "mean",
        aggr_demean: bool = False,
        cor_method: str = "correlation",
        cor_method_params: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        super().__init__(
            detrend=detrend,
            standardize=standardize,
            low_pass=low_pass,
            high_pass=high_pass,
            t_r=t_r,
            masks=masks,
            parcellation=parcellation,
            aggr_method=aggr_method,
            aggr_demean=aggr_demean,
            cor_method=cor_method,
            cor_method_params=cor_method_params,
            name=name,
        )

    def compute(self, input, extra_input):
        """Perform region-wise denoising and compute connectivity.
        1. Time series extraction
        2. Aggregation into region-wise time series
        3. Cleaning the region-wise signal
        4. FC generation

        Parameters
        ----------
        input: The input data.

        extra_input: Any additional input data.

        Returns
        -------
            Dict: A dictionary containing the computed connectivity data
            along with other metadata.

        """

        # Create dictionary for output
        out = {}

        # Start a timer
        t_start = datetime.now()

        # Perform ROI-level denoising
        voxelts = self._extract_timeseries(input, extra_input)
        aggregatets = self._aggregate(voxelts, input, extra_input)
        cleants = self._clean(aggregatets, input)

        # Save FC and processing time
        out["data"] = self.connectivity(cleants)

        proc_time = datetime.now() - t_start
        logger.info(f"Time from extraction to connectivity: {proc_time}")

        # Create column names
        out["row_names"] = self._labels
        out["col_names"] = self._labels
        out["matrix_kind"] = "tril"
        out["diagonal"] = False

        return out
