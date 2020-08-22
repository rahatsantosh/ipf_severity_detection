import glob
import shutil

destination_path = "../../data/interim/chest_xray/img/"
pattern = "../../data/external/chest_xray/img_folder/*/*/*"
for img in glob.glob(pattern):
    shutil.copy(img, destination_path)
