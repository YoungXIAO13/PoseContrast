import argparse
import os, sys
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle

import torch
from torch.utils.data import DataLoader

from model.resnet import resnet50
from dataset.vp_data import Pascal3D
from model.model_utils import load_checkpoint


import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import time



import argparse
parser = argparse.ArgumentParser()

# network training procedure settings
parser.add_argument('-n', '--nb', type=int, default=1)
parser.add_argument('-m', '--model', type=int, default=0, 
	help='index of models, 0: random; 1: MOCOv2;  2: PoseContrast')
parser.add_argument('-d', '--data', type=int, default=0, 
	help='index of datasets, 0: Pascal3D; 1: Pix3D;  2: ObjectNet3D')
parser.add_argument('-b', '--bin', type=int, default=10, 
	help='azimuth angle bin size for each cluster')
args = parser.parse_args()



os.environ["CUDA_VISIBLE_DEVICES"] = '0'

net_feat = resnet50(num_classes=128)
net_feat.cuda()

if args.model == 1:
	# ImageNet pre-trained model weights
	load_checkpoint(net_feat, '/home/xiao/Projects/PoseContrast/pretrain_models/res50_moco_v2_800ep_pretrain.pth')

if args.model == 2:
	# Pascal3D trained model weights
	state = torch.load('/home/xiao/Projects/PoseContrast/exps/PoseContrast_Pascal3D_MOCOv2/ckpt.pth')
	net_feat.load_state_dict(state['net_feat'])



##-------------------------------
if args.data == 0:
	root_dir = os.path.join('/home/xiao/Projects/TransferViewpoint/data/Pascal3D')
	annotation_file = 'Pascal3D.txt'

if args.data == 1:
	root_dir = os.path.join('/home/xiao/Projects/TransferViewpoint/data/Pix3D')
	annotation_file = 'Pix3D.txt'

if args.data == 2:
	root_dir = os.path.join('/home/xiao/Projects/TransferViewpoint/data/ObjectNet3D')
	annotation_file = 'ObjectNet3D.txt'

dataset_val = Pascal3D(train=False, root_dir=root_dir, annotation_file=annotation_file, offset=0)
data_loader = DataLoader(dataset_val, batch_size=64, shuffle=False, num_workers=4)

print(len(dataset_val))
##-------------------------------



##-------------------------------
net_feat.eval()
sample_feats = []
sample_labels = []

with torch.no_grad():
    for i, data in enumerate(data_loader):
        cls_index, im, label, im_flip = data
        im = im.cuda()
        feat, _ = net_feat(im)
        sample_feats.extend(feat.cpu().numpy())
        sample_labels.extend(label.numpy())

sample_feats = np.array(sample_feats)
feats = sample_feats.reshape(-1, sample_feats.shape[-1])

sample_labels = np.array(sample_labels)
labels = sample_labels.reshape(-1, sample_labels.shape[-1])

print(feats.shape, labels.shape)
##-------------------------------



##-------------------------------
X = feats
y = labels[:, 0] // args.bin

feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['y'] = y
df['label'] = df['y'].apply(lambda i: str(i))

X, y = None, None

print('Size of the dataframe: {}'.format(df.shape))


df_subset = df.copy()
data_subset = df_subset[feat_cols].values
##-------------------------------


##-------------------------------
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df[feat_cols].values)
# df['pca-one'] = pca_result[:,0]
# df['pca-two'] = pca_result[:,1] 
# df['pca-three'] = pca_result[:,2]
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(df)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# df['tsne-2d-one'] = tsne_results[:,0]
# df['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     palette=sns.color_palette("hls", int(360 / args.bin)),
#     data=df,
#     legend="full",
#     alpha=0.3
# )
##-------------------------------



##-------------------------------
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(data_subset)
print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_50.explained_variance_ratio_)))

time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
tsne_pca_results = tsne.fit_transform(pca_result_50)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

df_subset['tsne-pca50-one'] = tsne_pca_results[:,0]
df_subset['tsne-pca50-two'] = tsne_pca_results[:,1]
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-pca50-one", y="tsne-pca50-two",
    hue="y",
    palette=sns.color_palette("hls", int(360 / args.bin)),
    data=df_subset,
    legend="full",
    alpha=0.3
)


##-------------------------------
model_names = {0: 'Random', 1: 'MOCOv2', 2: 'PoseContrast'}
data_names = {0: 'Pascal3D', 1: 'Pix3D', 2: 'ObjectNet3D'}

fig_path = os.path.join('/home/xiao', 'vis_feat', '{}'.format(args.nb), 'bin{}'.format(args.bin),
	'tSNE_{}_{}_bin={}.png'.format(data_names[args.data], model_names[args.model], args.bin))
os.makedirs(os.path.dirname(fig_path), exist_ok=True)
plt.savefig(fig_path)


