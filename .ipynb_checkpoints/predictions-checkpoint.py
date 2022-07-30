#!/usr/bin/env python
# coding: utf-8
# %%


import timm
from fastai.vision.all import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob

root_dir = "/media/hdd/ART/Refs/"
new_path = Path(root_dir)/"temporary_acceptance"
predictions_path = Path("/home/eragon/Downloads/new_references")

def predict_batch(self, item, rm_type_tfms=None, with_input=False):
    dl = self.dls.test_dl(item, rm_type_tfms=rm_type_tfms, num_workers=0)
    ret = self.get_preds(dl=dl,with_input=False, with_decoded=True)
    return ret
Learner.predict_batch = predict_batch

learn = load_learner("export.pkl")

tst_files = get_image_files(predictions_path)

print(len(tst_files))

classes = learn.dls.vocab

preds = learn.predict_batch(tst_files)
preds_mapped = list(map(lambda x: classes[int(x)] , preds[2]))

for i,file in enumerate(tst_files):
    temp_path = new_path/preds_mapped[i]
    temp_path.mkdir(exist_ok=True, parents=True)
    shutil.move(file, temp_path)


# %%




