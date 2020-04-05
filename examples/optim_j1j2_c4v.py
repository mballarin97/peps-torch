import context
import torch
import argparse
import config as cfg
from ipeps.ipeps_c4v import *
from groups.pg import *
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v import ctmrg_c4v
from ctm.one_site_c4v.rdm_c4v import rdm2x1_sl
from ctm.one_site_c4v import transferops_c4v
from models import j1j2
# from optim.ad_optim_sgd_mod import optimize_state
from optim.ad_optim_lbfgs_mod import optimize_state
# from optim.ad_optim import optimize_state
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("-j1", type=float, default=1., help="nearest-neighbour coupling")
parser.add_argument("-j2", type=float, default=0., help="next nearest-neighbour coupling")
parser.add_argument("-top_freq", type=int, default=-1, help="freuqency of transfer operator spectrum evaluation")
parser.add_argument("-top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
parser.add_argument("-force_cpu", action='store_true', help="evaluate energy on cpu")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    torch.set_num_threads(args.omp_cores)
    torch.manual_seed(args.seed)

    model= j1j2.J1J2_C4V_BIPARTITE(j1=args.j1, j2=args.j2)
    energy_f= model.energy_1x1_lowmem

    # initialize the ipeps
    if args.instate!=None:
        state= read_ipeps_c4v(args.instate)
        if args.bond_dim > max(state.get_aux_bond_dims()):
            # extend the auxiliary dimensions
            state= extend_bond_dim(state, args.bond_dim)
        state.add_noise(args.instate_noise)
        # state.sites[(0,0)]= state.sites[(0,0)]/torch.max(torch.abs(state.sites[(0,0)]))
        state.sites[(0,0)]= state.site()/state.site().norm()
    elif args.opt_resume is not None:
        state= IPEPS_C4V(torch.tensor(0.))
        state.load_checkpoint(args.opt_resume)
    elif args.ipeps_init_type=='RANDOM':
        bond_dim = args.bond_dim
        A= torch.rand((model.phys_dim, bond_dim, bond_dim, bond_dim, bond_dim),\
            dtype=cfg.global_args.dtype, device=cfg.global_args.device)
        # A= make_c4v_symm(A)
        # A= A/torch.max(torch.abs(A))
        A= A/A.norm()
        state = IPEPS_C4V(A)
    else:
        raise ValueError("Missing trial state: -instate=None and -ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    print(state)
    
    @torch.no_grad()
    def ctmrg_conv_f(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=dict({"log": []})
        rdm2x1= rdm2x1_sl(state, env, force_cpu=ctm_args.conv_check_cpu)
        dist= float('inf')
        if len(history["log"]) > 0:
            dist= torch.dist(rdm2x1, history["rdm"], p=2).item()
        history["rdm"]=rdm2x1
        history["log"].append(dist)
        if dist<ctm_args.ctm_conv_tol or len(history["log"]) >= ctm_args.ctm_max_iter:
            log.info({"history_length": len(history['log']), "history": history['log']})
            return True, history
        return False, history

    state_sym= to_ipeps_c4v(state)
    ctm_env= ENV_C4V(args.chi, state_sym)
    init_env(state_sym, ctm_env)
    
    ctm_env, *ctm_log = ctmrg_c4v.run(state_sym, ctm_env, conv_check=ctmrg_conv_f)
    loss= energy_f(state_sym, ctm_env)
    obs_values, obs_labels= model.eval_obs(state_sym,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    def loss_fn(state, ctm_env_in, opt_context):
        ctm_args= opt_context["ctm_args"]
        opt_args= opt_context["opt_args"]
        # 0) preprocess
        # create a copy of state, symmetrize and normalize making all operations
        # tracked. This does not "overwrite" the parameters tensors, living outside
        # the scope of loss_fn
        state_sym= to_ipeps_c4v(state, normalize=True)

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            init_env(state_sym, ctm_env_in)

        # 1) compute environment by CTMRG
        ctm_env_out, *ctm_log= ctmrg_c4v.run(state_sym, ctm_env_in, 
            conv_check=ctmrg_conv_f, ctm_args=ctm_args)
        
        # 2) evaluate loss with converged environment
        loss = energy_f(state_sym, ctm_env_out, force_cpu=args.force_cpu)

        return (loss, ctm_env_out, *ctm_log)

    def _to_json(l):
        re=[l[i,0].item() for i in range(l.size()[0])]
        im=[l[i,1].item() for i in range(l.size()[0])]
        return dict({"re": re, "im": im})

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        if ("line_search" in opt_context.keys() and not opt_context["line_search"]) \
            or not "line_search" in opt_context.keys():
            state_sym= to_ipeps_c4v(state, normalize=True)
            epoch= len(opt_context["loss_history"]["loss"])
            loss= opt_context["loss_history"]["loss"][-1]
            obs_values, obs_labels = model.eval_obs(state_sym, ctm_env)
            print(", ".join([f"{epoch}",f"{loss}"]+[f"{v}" for v in obs_values]+\
                [f"{torch.max(torch.abs(state.site((0,0))))}"]))

            if args.top_freq>0 and epoch%args.top_freq==0:
                coord_dir_pairs=[((0,0), (1,0))]
                for c,d in coord_dir_pairs:
                    # transfer operator spectrum
                    print(f"TOP spectrum(T)[{c},{d}] ",end="")
                    l= transferops_c4v.get_Top_spec_c4v(args.top_n, state_sym, ctm_env)
                    print("TOP "+json.dumps(_to_json(l)))

    def post_proc(state, ctm_env, opt_context):
        symm, max_err= verify_c4v_symm_A1(state.site())
        # print(f"post_proc {symm} {max_err}")
        if not symm:
            # force symmetrization outside of autograd
            with torch.no_grad():
                symm_site= make_c4v_symm(state.site())
                # we **cannot** simply normalize the on-site tensors, as the LBFGS
                # takes into account the scale
                # symm_site= symm_site/torch.max(torch.abs(symm_site))
                state.sites[(0,0)].copy_(symm_site)

    # optimize
    # optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn, post_proc=post_proc)
    optimize_state(state, ctm_env, loss_fn, obs_fn=obs_fn)

    # compute final observables for the best variational state
    outputstatefile= args.out_prefix+"_state.json"
    state= read_ipeps_c4v(outputstatefile)
    ctm_env = ENV_C4V(args.chi, state)
    init_env(state, ctm_env)
    ctm_env, *ctm_log = ctmrg_c4v.run(state, ctm_env, conv_check=ctmrg_conv_f)
    opt_energy = energy_f(state,ctm_env)
    obs_values, obs_labels = model.eval_obs(state,ctm_env)
    print(", ".join([f"{args.opt_max_iter}",f"{opt_energy}"]+[f"{v}" for v in obs_values]))

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

class TestOpt(unittest.TestCase):
    def setUp(self):
        args.j2=0.0
        args.bond_dim=2
        args.chi=16
        args.opt_max_iter=3

    # basic tests
    def test_opt_SYMEIG(self):
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_opt_SYMEIG_gpu(self):
        args.GLOBALARGS_device="cuda:0"
        args.CTMARGS_projector_svd_method="SYMEIG"
        main()
