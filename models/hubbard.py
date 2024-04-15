import torch
import groups.su2 as su2
from ctm.generic import rdm
from ctm.one_site_c4v import rdm_c4v
from ctm.one_site_c4v import corrf_c4v
import config as cfg
from math import sqrt
import itertools
from ed_lgt.operators import Z2_FermiHubbard_dressed_site_operators, Z2_FermiHubbard_gauge_invariant_states
import numpy as np

def _cast_to_real(t):
    return t.real if t.is_complex() else t

class HUBBARD():
    def __init__(self, tunneling=1.0, onsite=8.0, penalty = 100.0, global_args=cfg.global_args):
        r"""
        :param hx: transverse field
        :param q: plaquette interaction
        :param global_args: global configuration
        :type hx: float
        :type q: float
        :type global_args: GLOBALARGS

        Build Ising Hamiltonian in transverse field with plaquette interaction

        .. math:: H = - \sum_{<i,j>} h2_{<i,j>} + q\sum_{p} h4_p - h_x\sum_i h1_i

        on the square lattice. Where the first sum runs over the pairs of sites `i,j`
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over
        all plaquettes `p`, and the last sum runs over all sites::

            y\x
               _:__:__:__:_
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
            ..._|__|__|__|_...
                :  :  :  :

        where

        * :math:`h2_{ij} = 4S^z_i S^z_j` with indices of h2 corresponding to :math:`s_i s_j;s'_i s'_j`
        * :math:`h4_p  = 16S^z_i S^z_j S^z_k S^z_l` where `i,j,k,l` labels the sites of a plaquette::

            p= i---j
               |   |
               k---l

          and the indices of `h4` correspond to :math:`s_is_js_ks_l;s'_is'_js'_ks'_l`

        * :math:`h1_i  = 2S^x_i`
        """
        self.dtype=global_args.torch_dtype
        self.device=global_args.device
        self.phys_dim=32

        self.tunneling = tunneling
        self.onsite = onsite
        self.penalty = penalty

        self.get_h()
        self.obs_ops = self.get_obs_ops()

    def get_h(self):
        dim = 2
        operators = Z2_FermiHubbard_dressed_site_operators(dim)
        basis, states = Z2_FermiHubbard_gauge_invariant_states(dim)

        projected_ops = {}
        for op in operators.keys():
            projected_ops[op] = basis["site"].transpose() @ operators[op] @ basis["site"]

        projected_ops["n_total"] = 0
        for s in "mp":
            for d in "xy":
                projected_ops["n_total"] += projected_ops[f"n_{s}{d}"]
        #projected_ops["id"] = np.identity(32)

        operators = {}
        for key, val in projected_ops.items():
            operators[key] = torch.Tensor(val.todense()).to(torch.complex128)
        operators["id"] = torch.eye(self.phys_dim, dtype=torch.complex128)


        # Local
        local = operators["N_pair_half"]

        # Hopping
        hopping_x_up = [operators["Qup_px_dag"], operators["Qup_mx"]]
        hopping_x_up_dag = [operators["Qup_px"], operators["Qup_mx_dag"]]
        hopping_y_up = [operators["Qup_py_dag"], operators["Qup_my"]]
        hopping_y_up_dag = [operators["Qup_py"], operators["Qup_my_dag"]]
        hopping_x_down = [operators["Qdown_px_dag"], operators["Qdown_mx"]]
        hopping_x_down_dag = [operators["Qdown_px"], operators["Qdown_mx_dag"]]
        hopping_y_down = [operators["Qdown_py_dag"], operators["Qdown_my"]]
        hopping_y_down_dag = [operators["Qdown_py"], operators["Qdown_my_dag"]]

        # Penalties
        # Plaquette penalty
        plaquette = [ operators[ii] for ii in ["C_px,py", "C_py,mx", "C_my,px", "C_mx,my"] ]
        # Link penalty
        local_p = operators["n_total"]
        hopping_x_p = [operators["n_px"], operators["n_mx"]]
        hopping_y_p = [operators["n_py"], operators["n_my"]]

        self.locals = [
            (self.onsite, local),
            (self.penalty, local_p)
        ]

        self.hopping_x = [
            (-1j*self.tunneling, hopping_x_up),
            (-1j*self.tunneling, hopping_x_down),
            (-1j*self.tunneling, hopping_x_up_dag),
            (-1j*self.tunneling, hopping_x_down_dag),
            (-2*self.penalty, hopping_x_p)
        ]

        self.hopping_y = [
            (-1j*self.tunneling, hopping_y_up),
            (-1j*self.tunneling, hopping_y_down),
            (-1j*self.tunneling, hopping_y_up_dag),
            (-1j*self.tunneling, hopping_y_down_dag),
            (-2*self.penalty, hopping_y_p)
        ]

        self.plaquettes = [
            (-self.penalty, plaquette),
            (self.penalty, [operators["id"] for _ in range(4)])
            ]

        self.local_obs = [
            ("S2", operators["S2"]),
            ("N_pair", operators["N_pair"]),
            ("N_tot", operators["N_tot"]),
        ]
        for op in ("n_px", "n_mx", "n_py", "n_my"):
            self.local_obs += [ (op, operators[op])]

        return

    def get_obs_ops(self):
        obs_ops = dict()
        s2 = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"]= 2*s2.SZ()
        obs_ops["sp"]= 2*s2.SP()
        obs_ops["sm"]= 2*s2.SM()
        return obs_ops

    def energy_1x1(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        For 1-site invariant iPEPS it's enough to construct a single reduced
        density matrix of a 2x2 plaquette. Afterwards, the energy per site `e` is
        computed by evaluating individual terms in the Hamiltonian through
        :math:`\langle \mathcal{O} \rangle = Tr(\rho_{2x2} \mathcal{O})`

        .. math::

            e = -(\langle h2_{<\bf{0},\bf{x}>} \rangle + \langle h2_{<\bf{0},\bf{y}>} \rangle)
            + q\langle h4_{\bf{0}} \rangle - h_x \langle h4_{\bf{0}} \rangle

        """

        # LD, RD, LU, RU
        rdm2x2= rdm.rdm2x2_uncontracted((0,0),state,env)

        # Normalize the state
        norm = self.sandwitch(rdm2x2)
        rdm2x2[0] /= norm

        elocal = 0
        for strength, op in self.locals:
            tmp = torch.einsum("ijkl,mk", (rdm2x2[0]), op)
            elocal += strength*self.sandwitch([tmp] + (rdm2x2[1:]) )

        ehopping_x = 0
        for strength, op in self.hopping_x:
            ehopping_lr0 = torch.einsum("ijkl, mk", rdm2x2[0], op[0])
            ehopping_lr1 = torch.einsum("ijkl, mk", rdm2x2[1], op[1])
            ehopping_x += strength*self.sandwitch([ehopping_lr0, ehopping_lr1] + rdm2x2[2:] )

        ehopping_y = 0
        for strength, op in self.hopping_y:
            ehopping_du0 = torch.einsum("ijkl, mk", rdm2x2[0], op[0])
            ehopping_du1 = torch.einsum("ijkl, mk", rdm2x2[2], op[1])
            ehopping_y += strength*self.sandwitch([ehopping_du0, rdm2x2[1], ehopping_du1, rdm2x2[3]] )

        eplaq = 0
        for strength, op in self.plaquettes:
            tmp = []
            for ii in range(4):
                tmp += [torch.einsum("ijkl, mk", rdm2x2[ii], op[ii]) ]

            eplaq = strength*self.sandwitch(tmp )

        energy_per_site= _cast_to_real(elocal+ehopping_x+ehopping_y+eplaq)

        return energy_per_site

    def sandwitch(self, this):

        temps = []
        for tt in (this):
            temps.append(
                torch.einsum("ijkk", tt)
            )
        down = torch.einsum("ij,kj", temps[0], temps[1])
        up = torch.einsum("ij,jk", temps[2], temps[3])
        res = torch.einsum("ij, ij", down, up)

        return res

    def eval_obs(self,state,env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. :math:`\langle 2S^z \rangle,\ \langle 2S^x \rangle` for each site in the unit cell

        """
        obs = {}
        with torch.no_grad():
            for coord, site in state.sites.items():
                rdm1x1= rdm.rdm1x1(coord, state, env)
                for label, op in self.local_obs:
                    obs[f"{label}"]= torch.trace(rdm1x1@op)

        # prepare list with labels and values
        obs_labels = list(obs.keys())
        obs_values = list(obs.values())
        obs_values = [ _cast_to_real(o) for o in obs_values]
        return obs_values, obs_labels
