import argparse

def get_args():
    parser = argparse.ArgumentParser()

    # env
    parser.add_argument('--env', type=str, default='CleanUp')
    parser.add_argument('--controller', type=str, default='OSC_POSE')
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument("--camera-names", type=str, nargs='+', default=['agentview', 'robot0_eye_in_hand'])
    parser.add_argument("--camera-height", type=int, default=84)
    parser.add_argument("--camera-width", type=int, default=84)

    # skill controller
    parser.add_argument('--primitive-set', type=str, nargs='+')
    parser.add_argument('--output-mode', type=str, default='max')
    parser.add_argument('--num-data-workers', type=int, default=40)

    # data collection
    parser.add_argument('--collect-demos', action='store_true', default=False)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--num-trajs', type=int)
    parser.add_argument('--num-primitives', type=int, default=50)
    parser.add_argument('--save', action='store_true', default=False)

    # data reformat
    parser.add_argument('--reformat-rollout-data', action='store_true', default=False)
    parser.add_argument("--num-others-per-traj", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.)
    parser.add_argument('--policy-pretrain', action='store_true', default=False)

    # model
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--idm-type-model-path", type=str, default=None)
    parser.add_argument("--idm-params-model-path", type=str, default=None)

    # trajectory parser
    parser.add_argument('--segment-demos', action='store_true', default=False)
    parser.add_argument("--demo-path", type=str, default=None)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument("--save-failed-trajs", action='store_true', default=False)
    parser.add_argument("--max-primitive-horizon", default=100, type=int)
    parser.add_argument('--segmented-data-dir', type=str, default=None)
    parser.add_argument('--parser-algo', type=str, default='dp')
    parser.add_argument("--playback-segmented-trajs", action='store_true', default=False)
    parser.add_argument("--num-augmentation-type", default=50, type=int)
    parser.add_argument("--num-augmentation-params", default=100, type=int)

    # policy evaluation
    parser.add_argument('--policy-type-path', type=str, default=None)
    parser.add_argument('--policy-params-path', type=str, default=None)
    parser.add_argument("--env-horizon", default=1000, type=int)
    parser.add_argument("--num-rollouts", default=50, type=int)

    # visualization
    parser.add_argument("--write-video", action='store_true', default=False)
    parser.add_argument("--verbose", action='store_true', default=False)


    parser.add_argument("--policy-feature-size", default=256, type=int)
    parser.add_argument("--policy-hidden-size", default=256, type=int)

    parser.add_argument("--proposal-feature-size", default=512, type=int)
    parser.add_argument("--proposal-hidden-size", default=512, type=int)
    parser.add_argument("--pt-load-path", default=None, type=str)
    parser.add_argument("--nheads", default=1, type=int)

    parser.add_argument("--model", type=str, default=None)

    parser.add_argument("--recognition-feature-size", default=64, type=int)
    parser.add_argument("--recognition-hidden-size", default=64, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("--dropout", default=0., type=float)

    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--lr", default=1e-1, type=float)
    parser.add_argument("--weight-decay", default=1e-3, type=float)
    parser.add_argument("--grad-clip", default=2, type=float)
    parser.add_argument("--num-epoch", default=50, type=int)
    parser.add_argument("--sequence-length", default=125, type=int)
    parser.add_argument('--cls-dataset', type=str, default=None)
    parser.add_argument('--policy-dataset', type=str, default=None)
    parser.add_argument('--optim', type=str, default=None)
    parser.add_argument("--label-smooth", default=0., type=float)
    parser.add_argument('--share-feature', action='store_true', default=False)
    parser.add_argument("--args-max-dim", default=7, type=int)

    parser.add_argument("--save-tb", action='store_true', default=False)
    parser.add_argument("--save-ckpt", action='store_true', default=False)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--log-interval", default=5, type=int)
    parser.add_argument("--save-dir", default="/home/tiangao/projects/primitives/exp_data", type=str)
    parser.add_argument("--policy-type", default=None, type=str)

    parser.add_argument("--pad-label", default=-1, type=int)
    parser.add_argument("--pad-input", default=0, type=int)

    parser.add_argument('--seed', type=int)
    parser.add_argument('--only-seg', action='store_true', default=False)
    parser.add_argument('--prob-thres', type=float, default=0.8)
    parser.add_argument('--image-input', action='store_true', default=False)
    parser.add_argument('--pred-args', action='store_true', default=False)
    parser.add_argument('--primitive-horizon', type=int, default=300)
    parser.add_argument('--alter-num', default=None)
    parser.add_argument('--p-filepath', type=str, default=None)
    parser.add_argument('--obs-aug', action='store_true', default=False)
    parser.add_argument('--seg-obs-aug', action='store_true', default=False)
    parser.add_argument('--aug-num', type=int, default=None)
    parser.add_argument('--eval-acc-log', action='store_true', default=False)
    parser.add_argument('--eval-seg-log', action='store_true', default=False)
    parser.add_argument('--eval-seg-demo-path', type=str, default=None)


    parser.add_argument('--cls-model-path', type=str, default=None)
    parser.add_argument('--playback-filter', action='store_true', default=False)
    parser.add_argument('--onlymax', action='store_true', default=False)
    parser.add_argument('--args-hack', action='store_true', default=False)
    parser.add_argument('--hard-negative', action='store_true', default=False)
    parser.add_argument('--use-aff-center', action='store_true', default=False)
    parser.add_argument('--use-count-based', action='store_true', default=False)
    parser.add_argument('--skill-done', action='store_true', default=False)
    parser.add_argument("--nfactor", type=int)
    parser.add_argument('--has-obj-ind', action='store_true', default=False)


    parser.add_argument('--feature-extractor', type=str, default='mlp')
    parser.add_argument('--cls-mode', type=str, default=None)
    parser.add_argument('--mode', type=str, default=None)
    parser.add_argument('--object-centric', action='store_true', default=False)
    parser.add_argument('--aux-loss', action='store_true', default=False)
    parser.add_argument('--aux-weight', type=float, default=0.)
    parser.add_argument('--aux-unnorm', action='store_true', default=False)
    parser.add_argument('--target-p', type=str, default=None)
    parser.add_argument('--num-env', type=int, default=20)
    parser.add_argument('--segtraj-filename', type=str, default=None)
    parser.add_argument('--eval-mode', type=str, default=None)
    parser.add_argument('--eval-parallel', action='store_true', default=False)
    parser.add_argument('--obj-in-hand', action='store_true', default=False)

    parser.add_argument('--explore-p', type=float, default=0.)
    parser.add_argument('--pred-args-prior-model-path', type=str, default=None)
    parser.add_argument('--pretrained-skill-path', type=str, default=None)
    parser.add_argument('--playback', action='store_true', default=False)
    parser.add_argument('--robomimic-config', type=str, default=None)
    parser.add_argument('--p-policy-path', type=str, default=None)
    parser.add_argument('--pred-args-policy-path', type=str, default=None)
    parser.add_argument('--policy-path', type=str, default=None)
    parser.add_argument('--video-name', type=str, default=None)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--ndemo', type=int, default=0)
    parser.add_argument('--cma-opt', action='store_true', default=False)
    parser.add_argument('--balance', action='store_true', default=False)

    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--fine-tune", action='store_true', default=False)
    parser.add_argument('--acc-type', type=str, default='acc')
    parser.add_argument('--start-limit', action='store_true', default=False)


    parser.add_argument("--ntraj", type=int, default=None)

    parser.add_argument("--aug-obs", action='store_true', default=False)
    parser.add_argument("--aug-seg-obs", action='store_true', default=False)

    args = parser.parse_args()
    return args


