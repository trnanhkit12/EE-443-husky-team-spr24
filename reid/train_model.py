import torchreid
from torchreid import models, utils
import torch

datamanager = torchreid.data.ImageDataManager(
    root='/content/gdrive/MyDrive/EE443/final_proj/dataset_for_reid_train',
    sources=['market1501'],
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=128
)
# osnet_ain_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/log/osnet_ain/events.out.tfevents.1716974331.cbea3be40bd7.33904.0'
# reid_model_ckpt = '/content/gdrive/MyDrive/EE443/final_proj/EE-443-husky-team-spr24/reid/osnet_x1_0_imagenet.pth'
# checkpoint = torch.load(osnet_ain_model_ckpt)
# model_osnet = checkpoint['model']

model_osnet_ain = models.build_model(
                            name='osnet_ain_x1_0',
                            num_classes=43,
                            pretrained=True,
                            loss="softmax")
model_osnet_ain = model_osnet_ain.cuda()

optimizer = torchreid.optim.build_optimizer(
    model_osnet_ain,
    optim='adam',
    staged_lr = True,
    lr=0.001,
    new_layers=['fc','classifier'],
    base_lr_mult = 10
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=2,
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager, model_osnet_ain, optimizer, scheduler=scheduler,
)

engine.run(
    save_dir='log/osnet_ain',            # TODO: fix this for drive
    max_epoch=12,
    eval_freq=6,
    print_freq=6,
    test_only=False,
)
print('test')
engine.run(
    save_dir='log/osnet_ain',            # TODO: fix this for drive
    max_epoch=10,
    eval_freq=5,
    print_freq=5,
    test_only=False,
)