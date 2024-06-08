import torchreid
from torchreid import models, utils
import torch

datamanager = torchreid.data.ImageDataManager(
    root='/content/gdrive/MyDrive/EE443/final_proj/dataset_for_reid_train',
    sources=['market1501'],
    height=256,
    width=128,
    batch_size_train=16,
    batch_size_test=128,
    transforms = ['random_erase', 'color_jitter', 'random_patch', 'random_crop']    # data augmentation strategies
)

model_osnet_ain = models.build_model(
                            name='osnet_ain_x1_0',
                            num_classes=43,         # determined using from the train-test ratio
                            pretrained=True,
                            loss="softmax")
model_osnet_ain = model_osnet_ain.cuda()

optimizer = torchreid.optim.build_optimizer(
    model_osnet_ain,                            # For osnet_ain
    optim='adam',
    staged_lr = True,
    lr=0.01,
    new_layers=['fc','classifier'],             # reinitiazlie both the classifier and fc
    base_lr_mult = 0.1
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=2,
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model_osnet_ain, optimizer, scheduler=scheduler,       # For osnet_ain
)

engine.run(
    save_dir='log/osnet_ain',        # TODO: for osnet_ain checkpoint saving dir
    max_epoch=10,
    eval_freq=5,
    print_freq=5,
    test_only=False,
)