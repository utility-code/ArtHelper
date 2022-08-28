from fastai.vision.all import *
from fastai.vision.widgets import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
from datetime import datetime

import click


@click.command()
@click.option("--root", default="/media/hdd/ART/Refs/SortedReferences", help="Where to save the images")
@click.option("--epochs", default=1, help="No of epochs", type=int)
def mainrunner(root, epochs):
    root_dir = root
    path = Path(root_dir)

    files = glob.glob(str(path) + "*/*/*")
#      print("Verifying images")
    #  failed = verify_images(files)
    #  failed = [Path.unlink(Path(x)) for x in failed]
    #  print("Verified")

    fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
                       get_items=get_image_files,
                       get_y=parent_label,
                       splitter=RandomSplitter(valid_pct=0.2, seed=42),
                       item_tfms=RandomResizedCrop(224, min_scale=0.5),
                       batch_tfms=aug_transforms(),
                       )
    dls = fields.dataloaders(path, bs=5)

    learn = load_learner("export.pkl")
    learn.dls = dls
    print("Loaded previous export")

    learn.fine_tune(epochs, base_lr=0.001)
    learn.export(f"{str(datetime.now())}")


if __name__ == '__main__':
    mainrunner()
