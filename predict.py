from fastai.vision.all import *
import click
from tqdm import tqdm


@click.command()
@click.option("--root", default="/media/hdd/ART/Refs/", help="Where to save the images")
@click.option("--output", default="/home/eragon/Downloads/new_references", help="New path")
def mainrunner(root, output):
    root_dir = Path(root)
    # (DO NOT CHANGE) The results will be stored in a subfolder in your references folder with this name
    new_path = Path(root_dir) / "temporary_acceptance"
    predictions_path = Path(output)
    Learner.predict_batch = predict_batch

    learn = load_learner("export.pkl")

    tst_files = get_image_files(predictions_path)

    classes = learn.dls.vocab

    preds = learn.predict_batch(tst_files)
    preds_mapped = list(map(lambda x: classes[int(x)], preds[2]))

    for i, file in tqdm(enumerate(tst_files), total=len(tst_files)):
        temp_path = new_path / preds_mapped[i]
        temp_path.mkdir(exist_ok=True, parents=True)
        shutil.move(file, temp_path)


def predict_batch(self, item, rm_type_tfms=None, with_input=False):
    dl = self.dls.test_dl(item, rm_type_tfms=rm_type_tfms, num_workers=0)
    ret = self.get_preds(dl=dl, with_input=False, with_decoded=True)
    return ret

if __name__ == '__main__':
    mainrunner()