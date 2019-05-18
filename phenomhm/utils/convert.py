import numpy as np
from scipy import constants as ct
from scipy import stats


class Converter:

    def __init__(self, key_order, **kwargs):

        self.conversions = []

        if 'ln_mT' in key_order and 'mr' in key_order:
            # replace ln_mT and mr with m1 and m2 respectively
            self.ind_ln_mT = key_order.index('ln_mT')
            self.ind_mr = key_order.index('mr')
            self.conversions.append(self.ln_mT_mr)

        if 'ln_m1' in key_order:
            self.ind_ln_m1 = key_order.index('ln_m1')
            self.conversions.append(self.ln_m1)

        if 'ln_m2' in key_order:
            self.ind_ln_m2 = key_order.index('ln_m2')
            self.conversions.append(self.ln_m2)

        """
        if 'ln_mC' in key_order and 'mu' in key_order:
            self.ind_ln_mC = key_order.index('ln_mC')
            self.ind_mu = key_order.index('mu')
            self.conversions.append(self.ln_mC_mu)
        """

        if 'ln_distance' in key_order:
            self.ind_ln_distance = key_order.index('ln_distance')
            self.conversions.append(self.ln_distance)

        if 'ln_tRef'in key_order:
            self.ind_ln_tRef = key_order.index('ln_tRef')
            self.conversions.append(self.ln_tRef)

        if 'ln_fRef' in key_order:
            self.ind_ln_fRef = key_order.index('ln_fRef')
            self.conversions.append(self.ln_fRef)

    def ln_m1(self, x):
        x[self.ind_ln_m1] = np.exp(x[self.ind_ln_m1])
        return x

    def ln_m2(self, x):
        x[self.ind_ln_m2] = np.exp(x[self.ind_ln_m2])
        return x

    def ln_mT_mr(self, x):
        mT = np.exp(x[self.ind_ln_mT])
        mr = x[self.ind_mr]

        m1 = mT/(1+mr)
        m2 = mT*mr/(1+mr)

        x[self.ind_ln_mT] = m1
        x[self.ind_mr] = m2
        return x

    """
    def ln_mC_mu(self, x):
        mC = np.exp(x[self.ind_ln_mC])
        mu = x[self.ind_mu]

        m1 = mT/(1+mr)
        m2 = mT*mr/(1+mr)

        x[self.ind_ln_mT] = m1
        x[self.ind_mr] = m2
        return x
    """

    def ln_distance(self, x):
        x[self.ind_ln_distance] = np.exp(x[self.ind_ln_distance])*1e9*ct.parsec  # Gpc
        return x

    def ln_tRef(self, x):
        x[self.ind_ln_tRef] = np.exp(x[self.ind_ln_tRef])
        return x

    def ln_fRef(self, x):
        x[self.ind_ln_fRef] = np.exp(x[self.ind_ln_fRef])
        return x

    def convert(self, x):
        for func in self.conversions:
            x = func(x)
        return x


class Recycler:

    def __init__(self, test_inds, key_order, **kwargs):

        # Setup of recycler
        key_order = [key_order[ind] for ind in test_inds]
        self.recycles = []
        if 'inc' in key_order:
            self.ind_inc = key_order.index('inc')
            self.recycles.append(self.inc)

        if 'lam' in key_order:
            # assumes beta is also there
            self.ind_lam = key_order.index('lam')
            self.ind_beta = key_order.index('beta')
            self.recycles.append(self.lam_and_beta)

        if 'phiRef' in key_order:
            self.ind_phiRef = key_order.index('phiRef')
            self.recycles.append(self.phiRef)

        if 'psi' in key_order:
            self.ind_psi = key_order.index('psi')
            self.recycles.append(self.psi)

    def inc(self, x):
        if x[self.ind_inc] < -np.pi/2. or x[self.ind_inc] > np.pi/2.:
            import pdb; pdb.set_trace()
            if x[self.ind_inc] < -np.pi/2:
                factor = 1.
            else:
                factor = -1.
            while (x[self.ind_inc] > np.pi/2 or x[self.ind_inc] < -np.pi/2):
                x[self.ind_inc] += factor*np.pi

        return x

    def lam_and_beta(self, x):
        if x[self.ind_lam] < 0.0 or x[self.ind_lam] > 2*np.pi:
            x[self.ind_lam] = x[self.ind_lam] % (2.*np.pi)

        if x[self.ind_beta] < -np.pi/2 or x[self.ind_beta] > np.pi/2:
            # assumes beta = 0 at ecliptic plane [-pi/2, pi/2]
            x_trans = np.cos(x[self.ind_beta])*np.cos(x[self.ind_lam])
            y_trans = np.cos(x[self.ind_beta])*np.sin(x[self.ind_lam])
            z_trans = np.sin(x[self.ind_beta])

            x[self.ind_lam] = np.arctan2(y_trans, x_trans)
            x[self.ind_beta] = np.arcsin(z_trans/np.sqrt(x_trans**2 + y_trans**2 + z_trans**2))  # check this with eccliptic coordinates

        return x

    def phiRef(self, x):
        if x[self.ind_phiRef] < 0.0 or x[self.ind_phiRef] > 2*np.pi:
            import pdb; pdb.set_trace()
            x[self.ind_phiRef] = x[self.ind_phiRef] % (2.*np.pi)
        return x

    def psi(self, x):
        if x[self.ind_psi] < 0.0 or x[self.ind_psi] > 2*np.pi:
            x[self.ind_psi] = x[self.ind_psi] % (2.*np.pi)
        return x

    def recycle(self, x):
        for func in self.recycles:
            x = func(x)
        return x
