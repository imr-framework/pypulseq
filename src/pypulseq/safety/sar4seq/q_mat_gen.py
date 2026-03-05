from __future__ import annotations

from .utils.gen_qpwr import gen_qpwr


def q_mat_gen(SAR_type: str, model: dict, qmat_write: int = 0):
    Ex = model['Ex']
    Ey = model['Ey']
    Ez = model['Ez']
    Tissue_types = model['Tissue_types']
    SigmabyRhox = model['SigmabyRhox']
    Mass_cell = model['Mass_cell']

    if SAR_type == 'Global':
        Qavg_df, Tissue_types, SigmabyRhox, Mass_cell, Mass_body, _ = gen_qpwr(
            Ex, Ey, Ez, Tissue_types, SigmabyRhox, Mass_cell, 'global', 'wholebody'
        )
        Qavg_tm = Qavg_df / float(Mass_body)

        Qavg_df_h, _, _, _, Mass_head, _ = gen_qpwr(
            Ex, Ey, Ez, Tissue_types, SigmabyRhox, Mass_cell, 'global', 'head'
        )
        Qavg_hm = Qavg_df_h / float(Mass_head)

        Q = {'Qtmf': Qavg_tm, 'Qhmf': Qavg_hm}
        if qmat_write:
            # Writing not ported yet (GUI path selection). Could be added with direct paths.
            pass
        return Q

    elif SAR_type == 'Local':
        # Not fully ported: would require get_Qavg and plotting
        raise NotImplementedError('Local SAR q-matrix generation not yet ported to Python.')

    else:
        raise ValueError("SAR_type must be 'Global' or 'Local'")



