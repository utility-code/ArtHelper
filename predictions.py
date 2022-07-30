#!/usr/bin/env python
# coding: utf-8
# %%


from fastai.vision.all import *

root_dir = "/media/hdd/ART/Refs/" # (CHANGE) The main folder where your references are stored
new_path = Path(root_dir)/"temporary_acceptance" # (DO NOT CHANGE) The results will be stored in a subfolder in your references folder with this name
predictions_path = Path("/home/eragon/Downloads/new_references") # (CHANGE) The folder where your unlabelled images are stored


def predict_batch(self, item, rm_type_tfms=None, with_input=False):
    dl = self.dls.test_dl(item, rm_type_tfms=rm_type_tfms, num_workers=0)
    ret = self.get_preds(dl=dl, with_input=False, with_decoded=True)
    return ret


Learner.predict_batch = predict_batch

learn = load_learner("export.pkl")

tst_files = get_image_files(predictions_path)

print(len(tst_files))

classes = learn.dls.vocab

preds = learn.predict_batch(tst_files)
preds_mapped = list(map(lambda x: classes[int(x)], preds[2]))

for i, file in enumerate(tst_files):
    temp_path = new_path/preds_mapped[i]
    temp_path.mkdir(exist_ok=True, parents=True)
    shutil.move(file, temp_path)


# %%
