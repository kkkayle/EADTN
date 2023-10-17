import argparse
from trainer import Trainer
parser = argparse.ArgumentParser(
        prog='DTI')
parser.add_argument('--seed', type=int, default=114514,help='')
parser.add_argument('--learning_rate', type=float, default=1e-4,help='')
parser.add_argument('--epoch', type=int, default=200,help='')
parser.add_argument('--batch_size', type=int, default=16,help='')
parser.add_argument('--patience', type=int, default=50,help='')
parser.add_argument('--decay_interval', type=int, default=10,help='')
parser.add_argument('--lr_decay', type=float, default=0.5,help='')
parser.add_argument('--weight_decay', type=float, default=1e-4,help='')
parser.add_argument('--embed_dim', type=int, default=64,help='')
parser.add_argument('--protein_kernel', type=list, default=[4,8,12],help='')
parser.add_argument('--drug_kernel', type=list, default=[4,6,8],help='')
parser.add_argument('--conv', type=int, default=40,help='')
parser.add_argument('--char_dim', type=int, default=64,help='')
parser.add_argument('--loss_epsilon', type=int, default=1,help='')
parser.add_argument('--k_fold',type=int,default=5,help='')
parser.add_argument('--fine_tuning_epochs', type=int, default=4,help='')
parser.add_argument('--dataset',default='Biosnap',choices=['BindingDB','Biosnap'],help='')
parser.add_argument('--scenario',type=str,default='Random',choices=['Random','Clustering'],help='')
parser.add_argument('--Clustering_basis',type=str,default='drug',choices=['drug','target'],help='')
args = parser.parse_args()



if __name__=='__main__':
    if args.scenario=='Random':
        trainer=Trainer(args)
        trainer.read_random_data()
        for fold in range(args.k_fold):
            loader_tuple=trainer.get_k_fold_dataloader(fold)
            trainer.train(loader_tuple,fold)
        trainer.test_random()
        
    elif args.scenario=='Clustering':
        trainer=Trainer(args)
        trainer.read_cluster_data()
        for fold in range(args.k_fold):
            loader_tuple=trainer.get_cluster_dataloader(fold)
            trainer.train(loader_tuple[:-1],fold)
            trainer.fine_tuning(loader_tuple[-1],fold)
        trainer.test_cluster_fine_tuning()
    