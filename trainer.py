# %%
from fastai.vision.all import *
from fastai.vision.widgets import *
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
# %%
root_dir = "/media/hdd/ART/Refs/SortedReferences"
path = Path(root_dir)
# %%
files = glob.glob(str(path)+"*/*/*")
# %%
# failed = verify_images(files)
# failed
# %%
# failed = [Path.unlink(x) for x in failed]
# %%
fields = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files,
                   get_y=parent_label,
                   splitter=RandomSplitter(valid_pct=0.2, seed=42),
                   item_tfms=RandomResizedCrop(224, min_scale=0.5),
                   batch_tfms=aug_transforms(),
                   )

# %%
dls = fields.dataloaders(path, bs=64)
# %%
dls.vocab
# %%
dls.show_batch()
# %%
learn = vision_learner(dls, resnet34, metrics=[error_rate, accuracy])
# %%
learn.fine_tune(4, base_lr=0.001)
# %%
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
# %%
cleaner = ImageClassifierCleaner(learn)
