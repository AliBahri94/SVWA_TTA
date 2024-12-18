#Pretraining      (Network has to be selected in pretrain_<dataset_name>.yaml)
CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_modelnet.yaml --dataset modelnet

CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_scanobject.yaml --dataset scanobject_nbg --ckpts models/pretrain.pth

CUDA_VISIBLE_DEVICES=0 python train.py --only_cls --config cfgs/pre_train/pretrain_shapenetcore.yaml --dataset shapenetcore

#TTT - Online
CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name modelnet --online --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_src_only_PointMAE.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name scanobject --online --config cfgs/tta/tta_scanobj.yaml --ckpts models/objectnn_src_only_PointMAE.pth

CUDA_VISIBLE_DEVICES=0 python ttt.py --dataset_name shapenetcore --online --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_src_only_PointMAE.pth


#Inference only
CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name modelnet --config cfgs/tta/tta_modelnet.yaml --ckpts models/modelnet_src_only.pth --test_source

CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name scanobject --config cfgs/tta/tta_scanobj.yaml --ckpts models/scanobject_src_only.pth --test_source

CUDA_VISIBLE_DEVICES=0 python test.py --dataset_name shapenetcore --config cfgs/tta/tta_shapenet.yaml --ckpts models/shapenet_src_only.pth --test_source
