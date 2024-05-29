import torchreid
from torchreid import models, utils
import torch

datamanager = torchreid.data.ImageDataManager(
    root='/content/gdrive/MyDrive/EE443/final_proj/dataset_for_reid_train',
    sources=['market1501'],
    height=256,
    width=128,
    batch_size_train=34,
    batch_size_test=128
)
# osnet_ain_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/log/osnet_ain/events.out.tfevents.1716974331.cbea3be40bd7.33904.0'
# reid_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'
# checkpoint = torch.load(osnet_ain_model_ckpt)
# model_osnet = checkpoint['model']

model_osnet_ain = models.build_model(
                            name='osnet_ain_x1_0',
                            num_classes=66,
                            pretrained=True,
                            loss="softmax")
model_osnet_ain = model_osnet_ain.cuda()

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
    stepsize=7
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model_osnet_ain, optimizer, margin=0.3,
    weight_t=0.7, weight_x=1, scheduler=scheduler
)

engine.run(
    save_dir='log/osnet_ain',            # TODO: fix this for drive
    max_epoch=30,
    eval_freq=10,
    print_freq=10,
    test_only=False,
)