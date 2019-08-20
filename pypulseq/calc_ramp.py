import numpy as np

from pypulseq.opts import Opts


def calc_ramp(k0: np.ndarray, k_end: np.ndarray, system: Opts = Opts(), max_points: int = 500, max_grad=np.zeros(0),
              max_slew=np.zeros(0)):
    """
    Join the points `k0` and `k_end` in three-dimensional  k-space in minimal time, observing the gradient and slew limits
    (`max_grad` and `max_slew` respectively), and the gradient strength G0 before `k0[:, 1]` and Gend after
    `k_end[:, 1]`. In the context of a fixed gradient dwell time this is a discrete problem with an a priori unknown
    number of discretization steps. Therefore this method tries out the optimization with 0 steps, then 1 step, and so
    on, until  all conditions can be fulfilled, thus yielding a short connection.

    Parameters
    ----------
    k0 : numpy.ndarray
        Two preceding points in k-space. Shape is `[3, 2]`. From these points, the starting gradient will be calculated.
    k_end : numpy.ndarray
        Two following points in k-space. Shape is `[3, 2]`. From these points, the target gradient will be calculated.
    system : Opts, optional
        System limits. Default is a system limits object initialised to default values.
    max_points : int, optional
        Maximum number of k-space points to be used in connecting `k0` and `k_end`. Default is 500.
    max_grad : float/list, optional
        Maximum total gradient strength. Either a single value or one value for each coordinate, of shape `[3, 1]`.
        Default is 0.
     max_slew : float/list, optional
        Maximum total slew rate. Either a single value or one value for each coordinate, of shape `[3, 1]`.
        Default is 0.

    Returns
    -------
    k_out : numpy.ndarray
        Connected k-space trajectory.
    success : bool
        Boolean flag indicating if `k0` and `k_end` were successfully joined.
    """

    def __inside_limits(grad, slew):
        if mode == 0:
            grad2 = np.sum(np.square(grad), axis=1)
            slew2 = np.sum(np.square(slew), axis=1)
            ok = np.all(np.max(grad2) <= np.square(max_grad)) and np.all(np.max(slew2) <= np.square(max_slew))
        else:
            ok = (np.sum(np.max(np.abs(grad), axis=1) <= max_grad) == 3) and (
                    np.sum(np.max(np.abs(slew), axis=1) <= max_slew) == 3)

        return ok

    def __joinleft0(k0, k_end, G0, G_end, use_points):
        if use_points == 0:
            G = np.stack((G0, (k_end - k0) / grad_raster, G_end)).T
            S = (G[:, 1:] - G[:, :-1]) / grad_raster

            k_out_left = np.zeros((3, 0))
            success = __inside_limits(G, S)

            return success, k_out_left

        dk = (k_end - k0) / (use_points + 1)
        kopt = k0 + dk
        Gopt = (kopt - k0) / grad_raster
        Sopt = (Gopt - G0) / grad_raster

        okGopt = np.sum(np.square(Gopt)) <= np.square(max_grad)
        okSopt = np.sum(np.square(Sopt)) <= np.square(max_slew)

        if okGopt and okSopt:
            k_left = kopt
        else:
            a = np.multiply(max_grad, grad_raster)
            b = np.multiply(max_slew, grad_raster ** 2)

            dkprol = G0 * grad_raster
            dkconn = dk - dkprol

            ksl = k0 + dkprol + dkconn / np.linalg.norm(dkconn) * b
            Gsl = (ksl - k0) / grad_raster
            okGsl = np.sum(np.square(Gsl)) <= np.square(max_grad)

            kgl = k0 + np.multiply(dk / np.linalg.norm(dk), a)
            Ggl = (kgl - k0) / grad_raster
            Sgl = (Ggl - G0) / grad_raster
            okSgl = np.sum(np.square(Sgl)) <= np.square(max_slew)

            if okGsl:
                k_left = ksl
            elif okSgl:
                k_left = kgl
            else:
                c = np.linalg.norm(dkprol)
                c1 = np.divide(np.square(a) - np.square(b) + np.square(c), (2 * c))
                h = np.sqrt(np.square(a) - np.square(c1))
                kglsl = k0 + np.multiply(c1, np.divide(dkprol, np.linalg.norm(dkprol)))
                projondkprol = (kgl * dkprol.T) * (dkprol / np.linalg.norm(dkprol))
                hdirection = kgl - projondkprol
                kglsl = kglsl + h * hdirection / np.linalg.norm(hdirection)
                k_left = kglsl

        success, k = __joinright0(k_left, k_end, (k_left - k0) / grad_raster, G_end, use_points - 1)
        if len(k) != 0:
            if len(k.shape) == 1:
                k = k.reshape((len(k), 1))
            if len(k_left.shape) == 1:
                k_left = k_left.reshape((len(k_left), 1))
            k_out_left = np.hstack((k_left, k))
        else:
            k_out_left = k_left

        return success, k_out_left

    def __joinleft1(k0, k_end, G0, G_end, use_points):
        if use_points == 0:
            G = np.stack((G0, (k_end - k0) / grad_raster, G_end))
            S = (G[:, 1:] - G[:, :-1]) / grad_raster

            k_out_left = np.zeros((3, 0))
            success = __inside_limits(G, S)

            return success, k_out_left

        k_left = np.zeros(3)

        dk = (k_end - k0) / (use_points + 1)
        kopt = k0 + dk
        Gopt = (kopt - k0) / grad_raster
        Sopt = (Gopt - G0) / grad_raster

        okGopt = np.abs(Gopt) <= max_grad
        okSopt = np.abs(Sopt) <= max_slew

        dkprol = G0 * grad_raster
        dkconn = dk - dkprol

        ksl = k0 + dkprol + np.multiply(np.sign(dkconn), max_slew) * grad_raster ** 2
        Gsl = (ksl - k0) / grad_raster
        okGsl = np.abs(Gsl) <= max_grad

        kgl = k0 + np.multiply(np.sign(dk), max_grad) * grad_raster ** 2
        Ggl = (kgl - k0) / grad_raster
        Sgl = (Ggl - G0) / grad_raster
        okSgl = np.abs(Sgl) <= max_slew

        for ii in range(3):
            if okGopt[ii] == 1 and okSopt[ii] == 1:
                k_left[ii] = kopt[ii]
            elif okGsl[ii] == 1:
                k_left[ii] = ksl[ii]
            elif okSgl[ii] == 1:
                k_left[ii] = kgl[ii]
            else:
                print('Unknown error')

        success, k = __joinright1(k_left, k_end, (k_left - k0) / grad_raster, G_end, use_points - 1)
        if len(k) != 0:
            if len(k.shape) == 1:
                k = k.reshape((len(k), 1))
            if len(k_left.shape) == 1:
                k_left = k_left.reshape((len(k_left), 1))
            k_out_left = np.hstack((k_left, k))
        else:
            k_out_left = k_left

        return success, k_out_left

    def __joinright0(k0, k_end, G0, G_end, use_points):
        if use_points == 0:
            G = np.stack((G0, (k_end - k0) / grad_raster, G_end)).T
            S = (G[:, 1:] - G[:, :-1]) / grad_raster

            k_out_right = np.zeros((3, 0))
            success = __inside_limits(G, S)

            return success, k_out_right

        dk = (k0 - k_end) / (use_points + 1)
        kopt = k_end + dk
        Gopt = (k_end - kopt) / grad_raster
        Sopt = (G_end - Gopt) / grad_raster

        okGopt = np.sum(np.square(Gopt)) <= np.square(max_grad)
        okSopt = np.sum(np.square(Sopt)) <= np.square(max_slew)

        if okGopt and okSopt:
            k_right = kopt
        else:
            a = np.multiply(max_grad, grad_raster)
            b = np.multiply(max_slew, grad_raster ** 2)

            dkprol = -G_end * grad_raster
            dkconn = dk - dkprol

            ksl = k_end + dkprol + dkconn / np.linalg.norm(dkconn) * b
            Gsl = (k_end - ksl) / grad_raster
            okGsl = np.sum(np.square(Gsl)) <= np.square(max_grad)

            kgl = k_end + np.multiply(dk / np.linalg.norm(dk), a)
            Ggl = (k_end - kgl) / grad_raster
            Sgl = (G_end - Ggl) / grad_raster
            okSgl = np.sum(np.square(Sgl)) <= np.square(max_slew)

            if okGsl:
                k_right = ksl
            elif okSgl:
                k_right = kgl
            else:
                c = np.linalg.norm(dkprol)
                c1 = np.divide(np.square(a) - np.square(b) + np.square(c), (2 * c))
                h = np.sqrt(np.square(a) - np.square(c1))
                kglsl = k_end + np.multiply(c1, np.divide(dkprol, np.linalg.norm(dkprol)))
                projondkprol = (kgl * dkprol.T) * (dkprol / np.linalg.norm(dkprol))
                hdirection = kgl - projondkprol
                kglsl = kglsl + h * hdirection / np.linalg.norm(hdirection)
                k_right = kglsl

        success, k = __joinleft0(k0, k_right, G0, (k_end - k_right) / grad_raster, use_points - 1)
        if len(k) != 0:
            if len(k.shape) == 1:
                k = k.reshape((len(k), 1))
            if len(k_right.shape) == 1:
                k_right = k_right.reshape((len(k_right), 1))
            k_out_right = np.hstack((k, k_right))
        else:
            k_out_right = k_right

        return success, k_out_right

    def __joinright1(k0, k_end, G0, G_end, use_points):
        if use_points == 0:
            G = np.stack((G0, (k_end - k0) / grad_raster, G_end))
            S = (G[:, 1:] - G[:, :-1]) / grad_raster

            k_out_right = np.zeros((3, 0))
            success = __inside_limits(G, S)

            return success, k_out_right

        k_right = np.zeros(3)

        dk = (k0 - k_end) / (use_points + 1)
        kopt = k_end + dk
        Gopt = (k_end - kopt) / grad_raster
        Sopt = (G_end - Gopt) / grad_raster

        okGopt = np.abs(Gopt) <= max_grad
        okSopt = np.abs(Sopt) <= max_slew

        dkprol = -G_end * grad_raster
        dkconn = dk - dkprol

        ksl = k_end + dkprol + np.multiply(np.sign(dkconn), max_slew) * grad_raster ** 2
        Gsl = (k_end - ksl) / grad_raster
        okGsl = np.abs(Gsl) <= max_grad

        kgl = k_end + np.multiply(np.sign(dk), max_grad) * grad_raster
        Ggl = (k_end - kgl) / grad_raster
        Sgl = (G_end - Ggl) / grad_raster
        okSgl = np.abs(Sgl) <= max_slew

        for ii in range(3):
            if okGopt[ii] == 1 and okSopt[ii] == 1:
                k_right[ii] = kopt[ii]
            elif okGsl[ii] == 1:
                k_right[ii] = ksl[ii]
            elif okSgl[ii] == 1:
                k_right[ii] = kgl[ii]
            else:
                print('Unknown error')

        success, k = __joinleft1(k0, k_right, G0, (k_end - k_right) / grad_raster, use_points - 1)
        if len(k) != 0:
            if len(k.shape) == 1:
                k = k.reshape((len(k), 1))
            if len(k_right.shape) == 1:
                k_right = k_right.reshape((len(k_right), 1))
            k_out_right = np.hstack((k, k_right))
        else:
            k_out_right = k_right

        return success, k_out_right

    # =========
    # MAIN FUNCTION
    # =========
    if np.all(np.where(max_grad <= 0)):
        max_grad = [system.max_grad]
    if np.all(np.where(max_slew <= 0)):
        max_slew = [system.max_slew]

    grad_raster = system.grad_raster_time

    if len(max_grad) == 1 and len(max_slew) == 1:
        mode = 0
    elif len(max_grad) == 3 and len(max_slew) == 3:
        mode = 1
    else:
        raise ValueError('Input value max grad or max slew in invalid format.')

    G0 = (k0[:, 1] - k0[:, 0]) / grad_raster
    G_end = (k_end[:, 1] - k_end[:, 0]) / grad_raster
    k0 = k0[:, 1]
    k_end = k_end[:, 0]

    success = 0
    k_out = np.zeros((3, 0))
    use_points = 0

    while success == 0 and use_points <= max_points:
        if mode == 0:
            if np.linalg.norm(G0) > max_grad or np.linalg.norm(G_end) > max_grad:
                break
            success, k_out = __joinleft0(k0, k_end, G0, G_end, use_points)
        else:
            if np.abs(G0) > np.abs(max_grad) or np.abs(G_end) > np.abs(max_grad):
                break
            success, k_out = __joinleft1(k0, k_end, G0, G_end, use_points)
        use_points += 1

    return k_out, success
