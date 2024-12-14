from types import SimpleNamespace
from typing import List

import numpy as np


def asc_to_acoustic_resonances(asc: dict) -> List[dict]:
    """
    Convert ASC dictionary from readasc to list of acoustic resonances

    Parameters
    ----------
    asc : dict
        ASC dictionary, see readasc

    Returns
    -------
    List[dict]
        List of acoustic resonances (specified by frequency and bandwidth fields).
    """
    if 'aflGCAcousticResonanceFrequency' in asc:
        freqs = asc['aflGCAcousticResonanceFrequency']
        bw = asc['aflGCAcousticResonanceBandwidth']
    else:
        freqs = asc['asGPAParameters'][0]['sGCParameters']['aflAcousticResonanceFrequency']
        bw = asc['asGPAParameters'][0]['sGCParameters']['aflAcousticResonanceBandwidth']

    return [{'frequency': f, 'bandwidth': b} for f, b in zip(freqs.values(), bw.values()) if f != 0]


def asc_to_hw(asc: dict, cardiac_model: bool = False) -> SimpleNamespace:
    """
    Convert ASC dictionary from readasc to SAFE hardware description.

    Parameters
    ----------
    asc : dict
        ASC dictionary, see readasc
    cardiac_model : bool
        Whether or not to read the cardiac stimulation model instead of the
        default PNS model (returns None if not available)

    Returns
    -------
    SimpleNamespace
        SAFE hardware description
    """
    hw = SimpleNamespace()

    if 'asCOMP' in asc and 'tName' in asc['asCOMP']:
        hw.name = asc['asCOMP']['tName']
    else:
        hw.name = 'unknown'

    if 'GradPatSup' in asc:
        asc_pns = asc['GradPatSup']['Phys']['PNS']
    else:
        asc_pns = asc

    if cardiac_model:
        if 'GradPatSup' in asc and 'CarNS' in asc['GradPatSup']['Phys']:
            asc_pns = asc['GradPatSup']['Phys']['CarNS']
        else:
            return None

    hw.x = SimpleNamespace()
    hw.x.tau1 = asc_pns['flGSWDTauX'][0]  # ms
    hw.x.tau2 = asc_pns['flGSWDTauX'][1]  # ms
    hw.x.tau3 = asc_pns['flGSWDTauX'][2]  # ms
    hw.x.a1 = asc_pns['flGSWDAX'][0]
    hw.x.a2 = asc_pns['flGSWDAX'][1]
    hw.x.a3 = asc_pns['flGSWDAX'][2]
    hw.x.stim_limit = asc_pns['flGSWDStimulationLimitX']  # T/m/s
    hw.x.stim_thresh = asc_pns['flGSWDStimulationThresholdX']  # T/m/s

    hw.y = SimpleNamespace()
    hw.y.tau1 = asc_pns['flGSWDTauY'][0]  # ms
    hw.y.tau2 = asc_pns['flGSWDTauY'][1]  # ms
    hw.y.tau3 = asc_pns['flGSWDTauY'][2]  # ms
    hw.y.a1 = asc_pns['flGSWDAY'][0]
    hw.y.a2 = asc_pns['flGSWDAY'][1]
    hw.y.a3 = asc_pns['flGSWDAY'][2]
    hw.y.stim_limit = asc_pns['flGSWDStimulationLimitY']  # T/m/s
    hw.y.stim_thresh = asc_pns['flGSWDStimulationThresholdY']  # T/m/s

    hw.z = SimpleNamespace()
    hw.z.tau1 = asc_pns['flGSWDTauZ'][0]  # ms
    hw.z.tau2 = asc_pns['flGSWDTauZ'][1]  # ms
    hw.z.tau3 = asc_pns['flGSWDTauZ'][2]  # ms
    hw.z.a1 = asc_pns['flGSWDAZ'][0]
    hw.z.a2 = asc_pns['flGSWDAZ'][1]
    hw.z.a3 = asc_pns['flGSWDAZ'][2]
    hw.z.stim_limit = asc_pns['flGSWDStimulationLimitZ']  # T/m/s
    hw.z.stim_thresh = asc_pns['flGSWDStimulationThresholdZ']  # T/m/s

    if 'asGPAParameters' in asc:
        hw.x.g_scale = asc['asGPAParameters'][0]['sGCParameters']['flGScaleFactorX']
        hw.y.g_scale = asc['asGPAParameters'][0]['sGCParameters']['flGScaleFactorY']
        hw.z.g_scale = asc['asGPAParameters'][0]['sGCParameters']['flGScaleFactorZ']
    else:
        print('Warning: Gradient scale factors not in ASC file: assuming 1/pi')
        hw.x.g_scale = 1 / np.pi
        hw.y.g_scale = 1 / np.pi
        hw.z.g_scale = 1 / np.pi

    return hw
