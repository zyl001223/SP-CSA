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
from enhanceModule import EnhanceCls

import models
import utils
from datasets.samplersFineTuning import CategoriesSampler


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
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--dalle_shot", type=int, default=5)
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

    # Dataset related parameters
    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--dataset",
        default="miniimagenet",
        type=str,
        choices=["miniimagenet", "tieredimagenet", "fc100", "cifar_fs"],
    )
    parser.add_argument(
        "--data_path",
        required=False,
        default="/root/autodl-tmp/",
        type=str,
    )

    # Checkpoint to load
    parser.add_argument(
        "--mdl_checkpoint_path",
        required=False,
        default="./vitCheckpoint/miniImageNet",
        choices=["/root/autodl-tmp/dcafl/vitCheckpoint/miniImageNet", "/root/autodl-tmp/dcafl/vitCheckpoint/tieredImageNet",
                 "/root/autodl-tmp/dcafl/vitCheckpoint/FC100", "/root/autodl-tmp/dcafl/vitCheckpoint/CIFAR-FS",],
        type=str,
        help="""Path to checkpoint of model to be loaded. Actual checkpoint given via chkpt_epoch.""",
    )
    parser.add_argument(
        "--chkpt_epoch",
        default=5500,
        type=int,
        help="""Number of epochs of pretrained 
                                    model to be loaded for evaluation. Irrelevant if metaft is selected.""",
    )

    # 从之前的checkpoints中加载模型
    parser.add_argument(
        "--fusion_checkpoint_path",
        required=False,
        default="",
        choices=["/root/autodl-tmp/dcafl/vitCheckpoint/miniImageNet/fusion", "/root/autodl-tmp/dcafl/vitCheckpoint/tieredImageNet/fusion",
                 "/root/autodl-tmp/dcafl/vitCheckpoint/FC100/fusion", "/root/autodl-tmp/dcafl/vitCheckpoint/CIFAR-FS/fusion",],
        type=str,
        help="""Path to checkpoint of model to be loaded. Actual checkpoint given via chkpt_epoch.""",
    )
    parser.add_argument(
        "--fusion_chkpt_epoch",
        default=7,
        type=int,
    )

    # Misc
    parser.add_argument(
        "--output_dir",
        default="./fineTuning",
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

    parser.add_argument(
        "--use_dalle",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--use_dalle_loss",
        default=True,
        type=bool,
    )
    parser.add_argument(
        "--use_patch_embeddings",
        default=True,
        type=bool,
    )

    # config

    return parser


def set_up_dataset(args):
    # Datasets and corresponding number of classes
    if args.dataset == "miniimagenet":
        # (Vinyals et al., 2016), (Ravi & Larochelle, 2017)
        # train num_class = 64
        from datasets.dataloaders.miniimagenet.miniimagenetFineTuning import (
            MiniImageNet as dataset,
        )
    elif args.dataset == "tieredimagenet":
        # (Ren et al., 2018)
        # train num_class = 351
        from datasets.dataloaders.tieredimagenet.tieredimagenetFineTuning import (
            tieredImageNet as dataset,
        )
    elif args.dataset == "fc100":
        # (Oreshkin et al., 2018) Fewshot-CIFAR 100 -- orig. images 32x32
        # train num_class = 60
        from datasets.dataloaders.fc100.fc100FineTuning import DatasetLoader as dataset
    elif args.dataset == "cifar_fs":
        # (Bertinetto et al., 2018) CIFAR-FS (100) -- orig. images 32x32
        # train num_class = 64
        from datasets.dataloaders.cifar_fs.cifar_fsFineTuning import DatasetLoader as dataset
    else:
        raise ValueError("Unknown dataset. Please check your selection!")
    return dataset


def get_embeddings(model, data, args):
    output_embeddings = model(data)

    cls_embedding = output_embeddings[:, :1]
    bs_cls, seq_len_cls, emb_dim_cls = (
        cls_embedding.shape[0],
        cls_embedding.shape[1],
        cls_embedding.shape[2],
    )
    cls_embedding = cls_embedding.reshape(
        args.n_way, -1, seq_len_cls, emb_dim_cls)
    train_cls_embedding = cls_embedding[:, args.dalle_shot:, :, :]
    dalle_cls_embedding = cls_embedding[:, :args.dalle_shot, :, :]
    emb_cls_support, emb_cls_query = (  # torch.Size([5, 5, 1, 384]) torch.Size([5, 15, 1, 384])
        train_cls_embedding[:, :args.k_shot, :, :],
        train_cls_embedding[:, args.k_shot:, :, :],
    )
    # print(dalle_cls_embedding.shape, emb_cls_support.shape, emb_cls_query.shape) torch.Size([5, 5, 1, 384]) torch.Size([5, 1, 1, 384]) torch.Size([5, 15, 1, 384])
    emb_cls_support = emb_cls_support.reshape(-1, seq_len_cls, emb_dim_cls)
    emb_cls_query = emb_cls_query.reshape(-1, seq_len_cls, emb_dim_cls)
    dalle_cls_embedding = dalle_cls_embedding.reshape(
        -1, seq_len_cls, emb_dim_cls)

    patch_embeddings = output_embeddings[:, 1:]
    bs_patch, seq_len_patch, emb_dim_patch = (
        patch_embeddings.shape[0],
        patch_embeddings.shape[1],
        patch_embeddings.shape[2],
    )
    patch_embeddings = patch_embeddings.reshape(
        args.n_way, -1, seq_len_patch, emb_dim_patch)
    # print(patch_embeddings.shape) torch.Size([5, 21, 196, 384])
    train_patch_embedding = patch_embeddings[:, args.dalle_shot:, :, :]
    dalle_patch_embedding = patch_embeddings[:, :args.dalle_shot, :, :]
    emb_patch_support, emb_patch_query = (
        train_patch_embedding[:, : args.k_shot, :, :],
        train_patch_embedding[:, args.k_shot:, :, :],
    )
    # print(dalle_patch_embedding.shape,
    #       emb_patch_support.shape, emb_patch_query.shape) torch.Size([5, 5, 196, 384]) torch.Size([5, 1, 196, 384]) torch.Size([5, 15, 196, 384])
    emb_patch_support = emb_patch_support.reshape(
        -1, seq_len_patch, emb_dim_patch)
    emb_patch_query = emb_patch_query.reshape(-1, seq_len_patch, emb_dim_patch)
    dalle_patch_embedding = dalle_patch_embedding.reshape(
        -1, seq_len_patch, emb_dim_patch)

    return emb_cls_support, emb_cls_query, dalle_cls_embedding, emb_patch_support, emb_patch_query, dalle_patch_embedding


def distance_module(prototypes, cls_with_weighted_sums):
    prototypes_expanded = prototypes.unsqueeze(
        1).repeat(1, 75, 1)  # (5, 75, 384)
    distances = torch.norm(
        cls_with_weighted_sums - prototypes_expanded, dim=2)  # (5, 75)
    distances_t = distances.t()
    softmax_distances = F.softmax(-distances_t, dim=1)  # (75, 5)
    return softmax_distances


def run_validation(model, fusion_module, val_loader, args, epoch):
    model.eval()
    fusion_module.eval()

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

            support_set_vectors, query_set_vectors, dalle_emb_support, emb_patch_support, emb_patch_query, dalle_patch_embedding = get_embeddings(
                model, data, args
            )

            enhance_prototypes, fusion_cls = fusion_module(
                support_set_vectors, query_set_vectors, dalle_emb_support, emb_patch_support, emb_patch_query, dalle_patch_embedding)

            query_pred_logits = distance_module(enhance_prototypes, fusion_cls)

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

    DataSet = set_up_dataset(args)
    train_dataset = DataSet('train', args)
    train_sampler = CategoriesSampler(train_dataset.label, train_dataset.label_dalle, args.num_episodes_per_epoch, args.n_way,
                                      args.k_shot + args.query)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler,
                                               num_workers=args.num_workers, pin_memory=True)

    val_dataset = DataSet('val', args)
    val_sampler = CategoriesSampler(val_dataset.label, val_dataset.label_dalle, args.num_validation_episodes, args.n_way,
                                    args.k_shot + args.query)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler,
                                             num_workers=args.num_workers, pin_memory=True)

    print(f"\nTraining {args.n_way}-way {args.k_shot}-shot learning scenario.")

    # ============ Building model loading parameters from provided checkpoint ============
    # if the network is a vision transformer (i.e. vit_tiny, vit_small, ...)
    if args.arch in models.__dict__.keys():
        model = models.__dict__[args.arch](
            patch_size=args.patch_size,
            return_all_tokens=True
        )
    # Move model to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.args = args
    # Load weights from a checkpoint of the model to meta train -- Note that artifact information has to be adapted!
    if args.mdl_checkpoint_path:
        print('Loading model from provided path...')
        chkpt = torch.load(args.mdl_checkpoint_path +
                           f'/checkpoint{args.chkpt_epoch:04d}.pth')
        chkpt_state_dict = chkpt['teacher']
    # Adapt and load state dict into current model for evaluation
    model.load_state_dict(
        utils.match_statedict(chkpt_state_dict), strict=False)

    # Save args of loaded checkpoint model into folder for reference!
    try:
        with (Path(args.output_dir) / "args_checkpoint_pretrained_model.txt").open("w") as f:
            f.write(json.dumps(chkpt["args"].__dict__, indent=4))
    except:
        print("Arguments used during training not available, and will thus not be stored in eval folder.")

    fusion_module = EnhanceCls(384, 384, 384)
    fusion_module = fusion_module.to(device)
    if args.fusion_checkpoint_path:
        fusion_chkpt = torch.load(args.fusion_checkpoint_path +
                                  f'/fusion_module_checkpoint{args.fusion_chkpt_epoch:04d}.pth')
        fusion_chkpt_state_dict = fusion_chkpt['fusion_module']
        fusion_module.load_state_dict(fusion_chkpt_state_dict)

    optimizer = torch.optim.Adam([  
        {'params': model.parameters(), 'lr': 0.00001},  
        {'params': fusion_module.parameters(), 'lr': 0.0001}  
    ])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=50 * args.num_episodes_per_epoch,
        eta_min=0
    )

    # Partially based on DeepEMD data loading / labelling strategy:
    # label of query images  -- Note: the order of samples provided is AAAAABBBBBCCCCCDDDDDEEEEE...!
    label_query = torch.arange(args.n_way).repeat_interleave(args.query)
    label_query = label_query.type(torch.cuda.LongTensor)

    best_val_acc = 0.

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        fusion_module.train()
        train_tqdm_gen = tqdm(train_loader)

        train_ave_acc = utils.Averager()
        train_acc_record = np.zeros((args.num_episodes_per_epoch,))
        train_ave_loss = utils.Averager()
        train_loss_record = np.zeros((args.num_episodes_per_epoch,))
        ttl_num_batches = len(train_tqdm_gen)

        for i, batch in enumerate(train_tqdm_gen, 1):

            data, data_label = [_.cuda() for _ in batch]

            support_set_vectors, query_set_vectors, dalle_emb_support, emb_patch_support, emb_patch_query, dalle_patch_embedding = get_embeddings(
                model, data, args
            )

            enhance_prototypes, fusion_cls = fusion_module(
                support_set_vectors, query_set_vectors, dalle_emb_support, emb_patch_support, emb_patch_query, dalle_patch_embedding)

            query_pred_logits = distance_module(enhance_prototypes, fusion_cls)

            loss = F.cross_entropy(query_pred_logits, label_query)

            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
            lr_scheduler.step()  


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
        # Compute stats of finished epoch
        m, pm = utils.compute_confidence_interval(train_acc_record)
        m_loss, _ = utils.compute_confidence_interval(train_loss_record)
        result_list = ['Ep {} | Overall Train Loss {:.4f} | Train Acc {:.4f}'.format(epoch, train_ave_loss.item(),
                                                                                     train_ave_acc.item())]
        result_list.append(
            'Ep {} | Train Loss {:.4f} | Train Acc {:.4f} + {:.4f}'.format(epoch, m_loss, m, pm))
        print(result_list[1])

        print("Validating model...")
        val_acc, val_conf, val_loss = run_validation(
            model, fusion_module, val_loader, args, epoch)
        if val_acc > best_val_acc:

            torch.save({'teacher': model.state_dict(), 'epoch': epoch},
                       os.path.join(args.output_dir, f'checkpoint{epoch:04d}.pth'))
            torch.save({'fusion_module': fusion_module.state_dict(), 'epoch': epoch},
                       os.path.join(args.output_dir, f'fusion_module_checkpoint{epoch:04d}.pth'))
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

    non_essential_keys = ["num_workers", "output_dir", "data_path"]
    exp_hash = utils.get_hash_from_args(args, non_essential_keys)
    total_bs = str(total_bs).replace(":", "").replace("\\", "")
    out_dim = str(out_dim).replace(":", "").replace("\\", "")
    args.output_dir = os.path.join(
        args.output_dir,
        args.dataset + f"_{args.image_size}",
        args.arch,
        f"bs_{total_bs}",
        f"outdim_{out_dim}",
        exp_hash,
    )
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train(args)
