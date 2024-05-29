import torchreid
from torchreid import models, utils
import torch

datamanager = torchreid.data.ImageDataManager(
    root='/content/gdrive/MyDrive/EE443/final_proj/dataset_for_reid_train',
    sources=['market1501'],
    height=256,
    width=128,
    batch_size=32
)

reid_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'
checkpoint = torch.load(reid_model_ckpt)
model_osnet = checkpoint['model']

model_osnet_ain = models.build_model(name='osnet_ain_x1_0',
                            num_classes=40,
                            loss="softmax")

optimizer = torchreid.optim.build_optimizer(
    model_osnet_ain,
    optim='adam',
    lr=0.01,
    staged_lr=True,
    new_layers='classifier',
    base_lr_mult=0.1
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=20
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model_osnet_ain, optimizer
)

engine.run(
    save_dir='log/osnet_ain',            # TODO: fix this for drive
    max_epoch=60,
    eval_freq=10,
    print_freq=10,
    test_only=False,
)