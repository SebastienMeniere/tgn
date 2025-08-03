import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, HistoryEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics



torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=1, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate')
parser.add_argument('--patience', type=int, default=3, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=10, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=10, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=10, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=100, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)


def jitter_timestamps(ts_array, src_array, dst_array, seed=0,
                      max_jitter=5):      # seconds
    """
    Deterministically jitter each (src, dst, ts) by up to ±max_jitter/2 seconds,
    with different jitter per destination to preserve a reproducible order.
    """
    rng = np.random.RandomState(seed)
    # hash (src,dst,ts) into a reproducible pseudo-random in [-.5, .5]
    noise = rng.uniform(-0.5, 0.5, size=len(ts_array))
    return ts_array + noise * max_jitter

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA, different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)
# --------------------------------------------------------------------------- #
# 0.  Helpers & data
# --------------------------------------------------------------------------- #
node_features, edge_features, full_data, train_data, val_data, test_data, \
new_node_val_data, new_node_test_data = get_data(
    DATA,
    different_new_nodes_between_val_and_test=args.different_new_nodes,
    randomize_features=args.randomize_features)

def apply_jitter(df, seed=42, max_jitter=1):
    df.timestamps = jitter_timestamps(df.timestamps,
                                      df.sources,
                                      df.destinations,
                                      seed=seed,
                                      max_jitter=max_jitter)
    return df

for _df in (train_data, val_data, test_data, full_data):
    apply_jitter(_df)

train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder  = get_neighbor_finder(full_data,  args.uniform)

train_neg_sampler  = HistoryEdgeSampler(train_data.sources, train_data.destinations,
                                        train_data.timestamps)
val_neg_sampler    = HistoryEdgeSampler(full_data.sources,  full_data.destinations,
                                        full_data.timestamps, seed=0)
nn_val_neg_sampler = HistoryEdgeSampler(full_data.sources,  full_data.destinations,
                                        full_data.timestamps, seed=1)
test_neg_sampler   = HistoryEdgeSampler(full_data.sources,  full_data.destinations,
                                        full_data.timestamps, seed=2)
nn_test_neg_sampler = HistoryEdgeSampler(full_data.sources, full_data.destinations,
                                         full_data.timestamps, seed=3)

device = torch.device(f'cuda:{GPU}' if torch.cuda.is_available() else 'cpu')

mean_ts_src, std_ts_src, mean_ts_dst, std_ts_dst = compute_time_statistics(
        full_data.sources, full_data.destinations, full_data.timestamps)

# --------------------------------------------------------------------------- #
# 1.  Runs
# --------------------------------------------------------------------------- #
for run in range(args.n_runs):
    logger.info(f"=========== run {run} ===========")

    results_path = (f"results/{args.prefix}_{run}.pkl"
                    if run > 0 else f"results/{args.prefix}.pkl")
    Path("results/").mkdir(parents=True, exist_ok=True)

    # --------------- model -------------------------------------------------- #
    tgn = TGN(neighbor_finder=train_ngh_finder,
              node_features=node_features,
              edge_features=edge_features,
              device=device,
              n_layers=NUM_LAYER,
              n_heads=NUM_HEADS,
              dropout=DROP_OUT,
              use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM,
              memory_dimension=MEMORY_DIM,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_ts_src,
              std_time_shift_src=std_ts_src,
              mean_time_shift_dst=mean_ts_dst,
              std_time_shift_dst=std_ts_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep).to(device)

    optimizer   = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
    criterion   = torch.nn.BCELoss()
    early_stop  = EarlyStopMonitor(max_round=args.patience)

    n_inst      = len(train_data.sources)
    n_batch     = math.ceil(n_inst / BATCH_SIZE)
    logger.info(f"training instances: {n_inst}  |  batches/epoch: {n_batch}")

    # bookkeeping arrays
    val_aps, new_val_aps, epoch_times, train_losses = [], [], [], []

    # --------------- epochs ------------------------------------------------- #
    for epoch in range(NUM_EPOCH):
        t0 = time.time()
        if USE_MEMORY:
            tgn.memory.__init_memory__()

        tgn.set_neighbor_finder(train_ngh_finder)
        logger.info(f"epoch {epoch}")

        # -------- mini-batches -------------------------------------------- #
        for b_start in range(0, n_batch, args.backprop_every):
            optimizer.zero_grad()
            loss_acc = 0.0

            for step in range(args.backprop_every):
                b_idx = b_start + step
                if b_idx >= n_batch:
                    break

                s = b_idx * BATCH_SIZE
                e = min(n_inst, s + BATCH_SIZE)

                src  = train_data.sources[s:e]
                dst  = train_data.destinations[s:e]
                ts   = train_data.timestamps[s:e]
                eidx = train_data.edge_idxs[s:e]

                _, dst_neg = train_neg_sampler.sample(src, ts)

                y_pos = torch.ones(len(src), device=device)
                y_neg = torch.zeros(len(src), device=device)

                p_pos, p_neg = tgn.compute_edge_probabilities(
                                   src, dst, dst_neg, ts, eidx, NUM_NEIGHBORS)

                loss_acc += (criterion(p_pos.squeeze(), y_pos) +
                             criterion(p_neg.squeeze(), y_neg))

            loss = loss_acc / args.backprop_every
            loss.backward()
            optimizer.step()
            if USE_MEMORY:
                tgn.memory.detach_memory()

            print(f"batch {b_idx:05d} | loss {loss.item():.4f}")
            train_losses.append(loss.item())

        # --------------- validation --------------------------------------- #
        tgn.set_neighbor_finder(full_ngh_finder)
        if USE_MEMORY:
            mem_after_train = tgn.memory.backup_memory()

        val_ap, _ = eval_edge_prediction(tgn, val_neg_sampler,
                                         val_data, NUM_NEIGHBORS)

        if USE_MEMORY:
            mem_after_val = tgn.memory.backup_memory()
            tgn.memory.restore_memory(mem_after_train)

        nn_val_ap, _ = eval_edge_prediction(tgn, nn_val_neg_sampler,
                                            new_node_val_data, NUM_NEIGHBORS)

        if USE_MEMORY:
            tgn.memory.restore_memory(mem_after_val)

        val_aps.append(val_ap); new_val_aps.append(nn_val_ap)
        logger.info(f"val AP={val_ap:.4f}  |  new-node AP={nn_val_ap:.4f}")

        # --------------- early stop + checkpoint -------------------------- #
        ckpt_path = f"./saved_checkpoints/{args.prefix}-run{run}-epoch{epoch}.pth"
        torch.save(tgn.state_dict(), ckpt_path)

        if early_stop.early_stop_check(val_ap):
            best_epoch = early_stop.best_epoch
            best_path  = f"./saved_checkpoints/{args.prefix}-run{run}-epoch{best_epoch}.pth"
            tgn.load_state_dict(torch.load(best_path))
            tgn.eval()
            logger.info(f"early-stop → reload best epoch {best_epoch}")
            break

        epoch_times.append(time.time() - t0)

        # dump interim results
        pickle.dump({"val_aps": val_aps,
                     "new_val_aps": new_val_aps,
                     "train_losses": train_losses,
                     "epoch_times": epoch_times},
                    open(results_path, "wb"))

    # --------------- test -------------------------------------------------- #
    tgn.set_neighbor_finder(full_ngh_finder)
    if USE_MEMORY:
        mem_pre_test = tgn.memory.backup_memory()

    test_ap, _    = eval_edge_prediction(tgn, test_neg_sampler,
                                         test_data, NUM_NEIGHBORS)
    
    if USE_MEMORY:
            mem_after_val = tgn.memory.backup_memory()
            tgn.memory.restore_memory(mem_pre_test)
    
    nn_test_ap, _ = eval_edge_prediction(tgn, nn_test_neg_sampler,
                                         new_node_test_data, NUM_NEIGHBORS)
    
    if USE_MEMORY:
            tgn.memory.restore_memory(mem_after_val)

    logger.info(f"TEST AP={test_ap:.4f}  |  new-node AP={nn_test_ap:.4f}")

    # --------------- save final model ------------------------------------- #
    final_path = f"./saved_models/{args.prefix}-{args.data}-run{run}.pth"
    torch.save(tgn.state_dict(), final_path)
    logger.info(f"model saved → {final_path}")
