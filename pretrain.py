import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from pathlib import Path
from tqdm import tqdm

import models
import utils
from datasets.samplers import CategoriesSampler


def get_args_parser():
    parser = argparse.ArgumentParser("train", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
    )
    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
    )

    # FSL task/scenario related parameters
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument(
        "--query", type=int, default=15
    )
    parser.add_argument(
        "--set",
        type=str,
        default="train",
        choices=["train", "val", "test"],
    )

    parser.add_argument('--num_episodes_per_epoch', type=int, default=100,
                        help="""Number of episodes used for 1 epoch of meta fine tuning. """)
    parser.add_argument('--num_epochs', type=int, default=100,
                        help="""Maximum number of epochs for meta fine tuning. """)
    parser.add_argument('--num_validation_episodes', type=int, default=600,
                        help="""Number of episodes used for validation. """)

    # Optimisation loop parameters
    parser.add_argument('--meta_lr', type=float, default=0.0003,
                        help="""Learning rate for meta finetuning outer loop. """)
    parser.add_argument('--meta_momentum', type=float, default=0.9,
                        help="""Momentum for meta finetuning outer loop. """)
    parser.add_argument('--meta_weight_decay', type=float, default=0.0005,
                        help="""Weight decay for meta finetuning outer loop. """)
    parser.add_argument('--meta_lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'multistep'],
                        help="""Learning rate scheduler for meta finetuning outer loop.""")
    parser.add_argument('--meta_lr_step_size', type=int, default=20,
                        help="""Step size used for step or multi-step lr scheduler. Currently not really in use.""")

    # Dataset related parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="""Size of the squared input images, 224 for imagenet-style.""",
    )
    parser.add_argument(
        "--dataset",
        default="miniimagenet",
        type=str,
        choices=["miniimagenet", "tieredimagenet", "fc100", "cifar_fs"],
        help="Please specify the name of the dataset to be used for training.",
    )
    parser.add_argument(
        "--data_path",
        required=False,
        default="/root/autodl-tmp/",
        type=str,
        help="Please specify path to the root folder containing the training dataset(s). If dataset "
        "cannot be loaded, check naming convention of the folder in the corresponding dataloader.",
    )

    # Checkpoint to load
    parser.add_argument(
        "--mdl_checkpoint_path",
        required=False,
        default="./vitCheckpoint/miniImageNet",
        choices=["./vitCheckpoint/miniImageNet", "./vitCheckpoint/tieredImageNet",
                 "./vitCheckpoint/FC100", "./vitCheckpoint/CIFAR-FS",],
        type=str,
        help="""Path to checkpoint of model to be loaded. Actual checkpoint given via chkpt_epoch.""",
    )
    parser.add_argument(
        "--chkpt_epoch",
        default=1600,
        type=int,
        help="""Number of epochs of pretrained 
                                    model to be loaded for evaluation. Irrelevant if metaft is selected.""",
    )

    parser.add_argument(
        "--output_dir",
        default="./pretrain_second_results/",
        type=str,
        help="""Root path where to save correspondence images. 
                            If left empty, results will be stored in './eval_results/...'.""",
    )
    parser.add_argument("--seed", default=10, type=int,
                        help="""Random seed.""")
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="""Number of data loading workers per GPU.""",
    )

    return parser


def set_up_dataset(args):
    # Datasets and corresponding number of classes
    if args.dataset == "miniimagenet":
        # (Vinyals et al., 2016), (Ravi & Larochelle, 2017)
        # train num_class = 64
        from datasets.dataloaders.miniimagenet.miniimagenet import (
            MiniImageNet as dataset,
        )
    elif args.dataset == "tieredimagenet":
        # (Ren et al., 2018)
        # train num_class = 351
        from datasets.dataloaders.tieredimagenet.tieredimagenet import (
            tieredImageNet as dataset,
        )
    elif args.dataset == "fc100":
        # (Oreshkin et al., 2018) Fewshot-CIFAR 100 -- orig. images 32x32
        # train num_class = 60
        from datasets.dataloaders.fc100.fc100 import DatasetLoader as dataset
    elif args.dataset == "cifar_fs":
        # (Bertinetto et al., 2018) CIFAR-FS (100) -- orig. images 32x32
        # train num_class = 64
        from datasets.dataloaders.cifar_fs.cifar_fs import DatasetLoader as dataset
    else:
        raise ValueError("Unknown dataset. Please check your selection!")
    return dataset


def get_cls_embeddings(model, data, args):
    """Function to retrieve all patch embeddings of provided data samples, split into support and query set samples;
    Data arranged in 'aaabbbcccdddeee' fashion, so must be split appropriately for support and query set
    """
    # Forward pass through backbone model;
    # Important: This contains the [cls] token at position 0 ([:,0]) and the patch-embeddings after that([:,1:end]).
    cls_embedding = model(data)[:, :1]
    bs, seq_len, emb_dim = (
        cls_embedding.shape[0],
        cls_embedding.shape[1],
        cls_embedding.shape[2],
    )
    # shape[n way(5), k_shot(1|5) + query(15), seq_len(1), emb_dim(384)]
    cls_embedding = cls_embedding.view(args.n_way, -1, seq_len, emb_dim)
    emb_support, emb_query = (
        cls_embedding[:, : args.k_shot],
        cls_embedding[:, args.k_shot:],
    )
    return emb_support.reshape(-1, seq_len, emb_dim), emb_query.reshape(
        -1, seq_len, emb_dim
    )


def run_validation(model, val_loader, args, epoch):
    model.eval()
    # Create labels and loggers
    label_query = torch.arange(args.n_way).repeat_interleave(
        args.query)  # Samples arranged in an 'aabbccdd' fashion
    label_query = label_query.type(torch.cuda.LongTensor)
    val_ave_acc = utils.Averager()
    val_acc_record = np.zeros((args.num_validation_episodes,))
    val_ave_loss = utils.Averager()
    val_loss_record = np.zeros((args.num_validation_episodes,))
    val_tqdm_gen = tqdm(val_loader)
    # Run validation
    with torch.no_grad():
        for i, batch in enumerate(val_tqdm_gen, 1):
            data, _ = [_.cuda() for _ in batch]
            # Retrieve the patch embeddings for all samples, both support and query from Transformer backbone
            support_set_vectors, query_set_vectors = get_cls_embeddings(
                model, data, args
            )
            support_set_vectors = support_set_vectors.squeeze(
                1
            )
            prototypes = support_set_vectors.view(
                5, -1, support_set_vectors.size(-1)
            ).mean(dim=1).squeeze(1)

            query_set_vectors = query_set_vectors.squeeze(
                1
            )

            distances = torch.cdist(query_set_vectors, prototypes)
            neg_distances = -distances
            query_pred_logits = F.softmax(neg_distances, dim=1)

            loss = F.cross_entropy(query_pred_logits, label_query)

            val_acc = utils.count_acc(query_pred_logits, label_query) * 100
            val_ave_acc.add(val_acc)
            val_acc_record[i - 1] = val_acc
            m, pm = utils.compute_confidence_interval(val_acc_record[:i])
            val_ave_loss.add(loss)
            val_loss_record[i - 1] = loss
            m_loss, _ = utils.compute_confidence_interval(val_loss_record[:i])
            val_tqdm_gen.set_description(
                'Ep {} | batch {}: Loss epi:{:.2f} avg: {:.4f} | Acc: epi:{:.2f} avg: {:.4f}+{:.4f}'
                .format(epoch, i, loss, m_loss, val_acc, m, pm))
        # Compute stats of finished epoch
        m, pm = utils.compute_confidence_interval(val_acc_record)
        m_loss, _ = utils.compute_confidence_interval(val_loss_record)
        result_list = ['Ep {} | Overall Validation Loss {:.4f} | Validation Acc {:.4f}'
                       .format(epoch, val_ave_loss.item(), val_ave_acc.item())]
        result_list.append(
            'Ep {} | Validation Loss {:.4f} | Validation Acc {:.4f} + {:.4f}'.format(epoch, m_loss, m, pm))
        print(f'{result_list[1]}')
        # Return validation accuracy for this epoch
        return m, pm, m_loss


def train(args):
    utils.fix_random_seeds(args.seed)
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ Setting up dataset and dataloader ... ============
    DataSet = set_up_dataset(args)
    train_dataset = DataSet('train', args)
    train_sampler = CategoriesSampler(train_dataset.label, args.num_episodes_per_epoch, args.n_way,
                                      args.k_shot + args.query)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                               num_workers=args.num_workers, pin_memory=True)

    val_dataset = DataSet('val', args)
    val_sampler = CategoriesSampler(val_dataset.label, args.num_validation_episodes, args.n_way,
                                    args.k_shot + args.query)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler,
                                             num_workers=args.num_workers, pin_memory=True)

    print(
        f"\nPre-training {args.n_way}-way {args.k_shot}-shot learning scenario.")

    # ============ Building model loading parameters from provided checkpoint ============
    if args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True
        )
    else:
        raise ValueError(
            f"Unknown architecture: {args.arch}. Please choose one that is supported.")

    # Move model to GPU
    model = model.cuda()

    # Add arguments to model for easier access
    model.args = args

    # Load weights from a checkpoint of the model
    if args.mdl_checkpoint_path:
        print('Loading model from provided path...')
        chkpt = torch.load(args.mdl_checkpoint_path +
                           f'/checkpoint{args.chkpt_epoch:04d}.pth')
        chkpt_state_dict = chkpt['teacher']
    else:
        raise ValueError(
            "Checkpoint not provided or provided one could not be found.")

    # Adapt and load state dict into current model
    model.load_state_dict(
        utils.match_statedict(chkpt_state_dict), strict=False)

    # ============= Building the optimiser ==========================
    param_to_meta_learn = [{'params': model.parameters()}]
    meta_optimiser = torch.optim.SGD(param_to_meta_learn,
                                     lr=args.meta_lr,
                                     momentum=args.meta_momentum,
                                     nesterov=True,
                                     weight_decay=args.meta_weight_decay)
    if args.meta_lr_scheduler == 'step':
        meta_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            meta_optimiser,
            step_size=int(args.meta_lr_step_size),
            gamma=args.gamma
        )
    elif args.meta_lr_scheduler == 'multistep':
        meta_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            meta_optimiser,
            milestones=[int(_) for _ in args.meta_lr_step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.meta_lr_scheduler == 'cosine':  # default for our application
        meta_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            meta_optimiser,
            T_max=50 * args.num_episodes_per_epoch,
            eta_min=0
        )
    else:
        raise ValueError('No Such Scheduler')

    # Partially based on DeepEMD data loading / labelling strategy:
    # label of query images  -- Note: the order of samples provided is AAAAABBBBBCCCCCDDDDDEEEEE...!
    label_query = torch.arange(args.n_way).repeat_interleave(args.query)
    label_query = label_query.type(torch.cuda.LongTensor)

    best_val_acc = 0.

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_tqdm_gen = tqdm(train_loader)
        train_ave_acc = utils.Averager()
        train_acc_record = np.zeros((args.num_episodes_per_epoch,))
        train_ave_loss = utils.Averager()
        train_loss_record = np.zeros((args.num_episodes_per_epoch,))
        ttl_num_batches = len(train_tqdm_gen)

        for i, batch in enumerate(train_tqdm_gen, 1):
            data, label = [_.cuda() for _ in batch]
            # Retrieve the patch embeddings for all samples, both support and query from Transformer backbone
            support_set_vectors, query_set_vectors = get_cls_embeddings(
                model, data, args
            )

            support_set_vectors = support_set_vectors.squeeze(
                1
            )  # torch.Size([25, 384])

            prototypes = support_set_vectors.view(
                5, -1, support_set_vectors.size(-1)
            ).mean(dim=1).squeeze(1)  # torch.Size([5, 5, 384]) -> torch.Size([5, 1, 384]) -> torch.Size([5, 384])

            query_set_vectors = query_set_vectors.squeeze(
                1
            )  # torch.Size([75, 384])

            distances = torch.cdist(query_set_vectors, prototypes)

            query_pred_logits = F.softmax(-distances, dim=1)

            loss = F.cross_entropy(query_pred_logits, label_query)

            meta_optimiser.zero_grad()
            loss.backward()
            meta_optimiser.step()
            meta_lr_scheduler.step()

            train_acc = utils.count_acc(query_pred_logits, label_query) * 100
            train_ave_acc.add(train_acc)
            train_acc_record[i - 1] = train_acc
            m, pm = utils.compute_confidence_interval(train_acc_record[:i])
            train_ave_loss.add(loss)
            train_loss_record[i - 1] = loss
            m_loss, _ = utils.compute_confidence_interval(
                train_loss_record[:i])
            train_tqdm_gen.set_description(
                'Ep {} | bt {}/{}: Loss epi:{:.2f} avg: {:.4f} | Acc: epi:{:.2f} avg: {:.4f}+{:.4f}'.format(epoch, i,
                                                                                                            ttl_num_batches,
                                                                                                            loss,
                                                                                                            m_loss,
                                                                                                            train_acc,
                                                                                                            m, pm))
        m, pm = utils.compute_confidence_interval(train_acc_record)
        m_loss, _ = utils.compute_confidence_interval(train_loss_record)
        result_list = ['Ep {} | Overall Train Loss {:.4f} | Train Acc {:.4f}'.format(epoch, train_ave_loss.item(),
                                                                                     train_ave_acc.item())]
        result_list.append(
            'Ep {} | Train Loss {:.4f} | Train Acc {:.4f} + {:.4f}'.format(epoch, m_loss, m, pm))
        print(result_list[1])

        print("Validating model...")
        val_acc, val_conf, val_loss = run_validation(
            model, val_loader, args, epoch)
        if val_acc > best_val_acc:
            # Save model parameters
            torch.save({'teacher': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.output_dir, f'checkpoint{epoch:04d}.pth'))
            best_val_acc = val_acc
            print(f"Best validation acc: {val_acc} +- {val_conf}")
        print("Finished validation, running next epoch.\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("train", parents=[get_args_parser()])
    args = parser.parse_args()

    if args.mdl_checkpoint_path:
        out_dim = args.mdl_checkpoint_path.split("outdim_")[-1].split("/")[0]
        total_bs = args.mdl_checkpoint_path.split("bs_")[-1].split("/")[0]
    else:
        raise ValueError("No checkpoint provided!")

    args.__dict__.update({"out_dim": out_dim})
    args.__dict__.update({"batch_size_total": total_bs})

    if args.output_dir == "":
        args.output_dir = os.path.join(utils.get_base_path(), "train_results")

    non_essential_keys = ['num_workers', 'output_dir', 'data_path']
    exp_hash = utils.get_hash_from_args(args, non_essential_keys)
    args.output_dir = os.path.join(args.output_dir, args.dataset + f'_{args.image_size}', args.arch,
                                   f'ep_{args.chkpt_epoch+1}', f'bs_{total_bs}', f'outdim_{out_dim}', exp_hash)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(args)
