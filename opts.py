import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_classes', default=24, type=int, help='Number of classes (ucf101-24: 24, jhmdb-21: 21)')
    parser.add_argument('--dataset', default='ucf101-24', type=str, help='Select dataset from (ucf101-24, jhmdb-21)')
    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--end_epoch', default=25, type=int, help='Training ends at this epoch.')
    parser.add_argument('--resume_path', default='', type=str, help='Continue training from pretrained (.pth)')
    parser.add_argument('--data_cfg', default='', type=str, help='Configuration related to data')
    parser.add_argument('--cfg_file', default='', type=str, help='Configuration file')
    parser.add_argument('--backbone_3d', default='resnext101', type=str, help='(resnext101 | resnet101 | resnet50 | resnet18 | mobilenet_2x | mobilenetv2_1x | shufflenet_2x | shufflenetv2_2x')
    parser.add_argument('--backbone_3d_weights', default='', type=str, help='Load pretrained weights for 3d_backbone')
    parser.add_argument('--freeze_backbone_3d', action='store_true', help='If true, 3d_backbone is frozen, else it is finetuned.')
    parser.set_defaults(freeze_backbone_3d=False)
    parser.add_argument('--backbone_2d', default='darknet19', type=str, help='Currently there is only darknet19')
    parser.add_argument('--backbone_2d_weights', default='', type=str, help='Load pretrained weights for 3d_backbone')
    parser.add_argument('--freeze_backbone_2d', action='store_true', help='If true, 2d_backbone is frozen, else it is finetuned.')
    parser.set_defaults(freeze_backbone_2d=False)
    parser.add_argument('--evaluate', action='store_true', help='If true, model is not trained, but only evaluated.')
    parser.set_defaults(evaluate=False)

    args = parser.parse_args()
    return args