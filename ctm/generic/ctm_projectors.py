import torch
import config as cfg
from ipeps import IPEPS
from ctm.generic.env import ENV
from ctm.generic.ctm_components import *
from custom_svd import *

def ctm_get_projectors_4x4(direction, coord, state, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param direction: direction of the CTM move for which the projectors are to be computed
    :param coord: vertex (x,y) specifying (together with ``direction``) 4x4 tensor network 
                  used to build projectors
    :param state: wavefunction
    :param env: environment corresponding to ``state`` 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type direction: tuple(int,int) 
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor


    Compute a pair of projectors from two halfs of 4x4 tensor network given 
    by ``direction`` and ``coord``::

        Case of LEFT move <=> direction=(-1,0)
                                                _____________
        C--T---------------T---------------C = |_____R_______|
        T--A(coord)--------A(coord+(1,0))--T    |__|_____|__| 
        |  |               |               |   |_____Rt______|
        T--A(coord+(0,1))--A(coord+(1,1))--T    
        C--T---------------T---------------C

    This function constructs two halfs of a 4x4 network and then calls 
    :py:func:`ctm_get_projectors_from_matrices` for projector construction 
    """
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1):
        R, Rt = halves_of_4x4_CTM_MOVE_UP(coord, state, env, verbosity=verbosity)
    elif direction==(-1,0): 
        R, Rt = halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, verbosity=verbosity)
    elif direction==(0,1):
        R, Rt = halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, verbosity=verbosity)
    elif direction==(1,0):
        R, Rt = halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, verbosity=verbosity)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args)

def ctm_get_projectors_4x2(direction, coord, state, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param direction: direction of the CTM move for which the projectors are to be computed
    :param coord: vertex (x,y) specifying (together with ``direction``) 4x2 (vertical) or 
                  2x4 (horizontal) tensor network used to build projectors
    :param state: wavefunction
    :param env: environment corresponding to ``state`` 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type direction: tuple(int,int) 
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor


    Compute a pair of projectors from two enlarged corners making up 4x2 (2x4) tensor network 
    given by ``direction`` and ``coord``::

        Case of LEFT move <=> direction=(-1,0)
                                ____
        C--T---------------\ = |_R__|=\\
        T--A(coord)--------\\   |__|  ||
        |  |               ||  |_Rt_|=//
        T--A(coord+(0,1))--//    
        C--T---------------/

        Case of UP move <=> direction=(0,-1)
                                           ____    ___
        C--T---------T----------------C = |_Rt_|==|_R_|
        T--A(coord+(-1,0))--A(coord)--T    |  |    | |
        |  |         |                |     \_\===/_/
        \__\========/________________/

    This function constructs two enlarged corners of a 4x2 (2x4) network and then calls 
    :py:func:`ctm_get_projectors_from_matrices` for projector construction 
    """

    # function ctm_get_projectors_from_matrices expects first dimension of R, Rt
    # to be truncated. Instead c2x2 family of functions returns corners with 
    # index-position convention following the definition in env module 
    # (anti-clockwise from "up")
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1): # UP
        R= c2x2_RU(coord, state, env, verbosity=verbosity)
        Rt= c2x2_LU((coord[0]-1,coord[1]), state, env, verbosity=verbosity)
        Rt= torch.t(Rt)  
    elif direction==(-1,0): # LEFT
        R= c2x2_LU(coord, state, env, verbosity=verbosity)
        Rt= c2x2_LD((coord[0],coord[1]+1), state, env, verbosity=verbosity)
    elif direction==(0,1): # DOWN
        R= c2x2_LD(coord, state, env, verbosity=verbosity)
        R= torch.t(R)
        Rt= c2x2_RD((coord[0]+1,coord[1]), state, env, verbosity=verbosity)
        Rt= torch.t(Rt)
    elif direction==(1,0): # RIGHT
        R= c2x2_RD(coord, state, env, verbosity=verbosity)
        Rt= c2x2_RU((coord[0],coord[1]-1), state, env, verbosity=verbosity)
        Rt= torch.t(Rt)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args)

#####################################################################
# direction-independent function performing bi-diagonalization
#####################################################################

def ctm_get_projectors_from_matrices(R, Rt, chi, ctm_args=cfg.ctm_args, global_args=cfg.global_args):

    r"""
    :param R: tensor of shape (dim0, dim1)
    :param Rt: tensor of shape (dim0, dim1)
    :param chi: environment bond dimension
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type R: torch.tensor 
    :type Rt: torch.tensor
    :type chi: int
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor

    Given the two tensors R and Rt (R tilde) compute the projectors P, Pt (P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf)
        
        1. Perform SVD over :math:`R\widetilde{R}` contracted through index which is going to
           be truncated::
           
                       _______          ______
                dim1--|___R___|--dim0--|__Rt__|--dim1  ==SVD==> dim1(R)--U--S--V^+--dim1(Rt) 

           Hence, for the inverse :math:`(R\widetilde{R})^{-1}`::
              
                       ________          ________
                dim1--|__Rt^-1_|--dim0--|__R^-1__|--dim1 = dim1(Rt)--V--S^-1--U^+--dim1(R) 

        2. Approximate an identity :math:`RR^{-1}\widetilde{R}^{-1}\widetilde{R}` by truncating
           the result of :math:`SVD(R\widetilde{R}^{-1})`::

                           ____          ______          _______          ____
                I = dim0--|_R__|--dim1--|_R^-1_|--dim0--|_Rt^-1_|--dim1--|_Rt_|--dim0
                           ____          _____                            ____          ____
                I ~ dim0--|_R__|--dim1--|_U^+_|--St^-1/2--\chi--St^-1/2--|_V__|--dim1--|_Rt_|--dim0
        
           where :math:`\widetilde{S}` has been truncated to the leading :math:`\chi` singular values    
        
        3. Finally construct the projectors :math:`P, \widetilde{P}`::
                
                           ____          _____
                P = dim0--|_R__|--dim1--|_U^+_|--St^-1/2--\chi
                                     ____          ____
                Pt = \chi--St^-1/2--|_V__|--dim1--|_Rt_|--dim0

        The projectors :math:`P, \widetilde{P}` approximate contraction of the original
        matrices :math:`R, \widetilde{R}`::
                        
             _______     _________
            |___R___| ~ |___R_____|
             _|___|_      |     |
            |___Rt__|    dim0  dim1
                        __|___  |                                         
                        \_P__/  |
                          |     |
                         chi    |
                         _|__   |
                        /_Pt_\  |
                          |     |    
                         dim0  dim1
                         _|_____|_
                        |____Rt___|
    """
    assert R.shape == Rt.shape
    assert len(R.shape) == 2
    verbosity = ctm_args.verbosity_projectors

    #  SVD decomposition
    M = torch.mm(R.transpose(1, 0), Rt)
    U, S, V = truncated_svd_gesdd(M, chi) # M = USV^{T}

    # if abs_tol is not None: St = St[St > abs_tol]
    # if abs_tol is not None: St = torch.where(St > abs_tol, St, Stzeros)
    # if rel_tol is not None: St = St[St/St[0] > rel_tol]
    S = S[S/S[0] > ctm_args.projector_svd_reltol]
    S_zeros = torch.zeros((chi-S.size()[0]), dtype=global_args.dtype, device=global_args.device)
    S_sqrt = torch.rsqrt(S)
    S_sqrt = torch.cat((S_sqrt, S_zeros))
    if verbosity>0: print(S_sqrt)

    # Construct projectors
    # P = torch.einsum('i,ij->ij', S_sqrt, torch.mm(U.transpose(1, 0), R.transpose(1, 0)))
    P = torch.einsum('ij,j->ij', torch.mm(R, U), S_sqrt)
    # Pt = torch.einsum('i,ij->ij', S_sqrt, torch.mm(V.transpose(1, 0), Rt.transpose(1, 0)))
    Pt = torch.einsum('ij,j->ij', torch.mm(Rt, V), S_sqrt)

    return P, Pt