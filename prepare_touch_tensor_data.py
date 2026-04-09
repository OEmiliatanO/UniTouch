import glob
import random
import torch
import ImageBind.data as data
from ImageBind.models.x2touch_model_part import imagebind_huge, x2touch, ModalityType
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(0)

classes = [
    "sugar box", "tomato soup can", "mustard bottle", "bleach cleanser", 
    "mug", "power drill", "scissors", "adjustable wrench", "hammer", "baseball"
]
cls_to_idx = {cls: idx for idx, cls in enumerate(classes)}

prompts = [f"This feels like {cls}" for cls in classes]
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
imagebined_model = imagebind_huge(pretrained=True)
text_data = data.load_and_transform_text(prompts, device=device)
with torch.no_grad():
    imagebined_model = imagebined_model.to(device)
    text_features = imagebined_model({ModalityType.TEXT: text_data})[ModalityType.TEXT]  # Shape: [C, seq_len, 1024]
    eos_indices = text_data.argmax(dim=-1)
    text_features = text_features[torch.arange(text_features.shape[0]), eos_indices]
text_features = text_features.cpu()

torch.save(text_features, "YCB-Slide_text_features.pt")
root_dir = "/tmp3/Hans/YCB-Slide/dataset/"
dataset_len = 5
classes_touch_paths = {
    "sugar box" : [glob.glob(f"{root_dir}/real/004_sugar_box/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "tomato soup can" : [glob.glob(f"{root_dir}/real/005_tomato_soup_can/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "mustard bottle" : [glob.glob(f"{root_dir}/real/006_mustard_bottle/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "bleach cleanser" : [glob.glob(f"{root_dir}/real/021_bleach_cleanser/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "mug" : [glob.glob(f"{root_dir}/real/025_mug/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "power drill" : [glob.glob(f"{root_dir}/real/035_power_drill/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "scissors" : [glob.glob(f"{root_dir}/real/037_scissors/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "adjustable wrench" : [glob.glob(f"{root_dir}/real/042_adjustable_wrench/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "hammer" : [glob.glob(f"{root_dir}/real/048_hammer/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)],
    "baseball" : [glob.glob(f"{root_dir}/real/055_baseball/dataset_{dataset_idx}/frames/*.jpg") for dataset_idx in range(dataset_len)]
}
all_labels = []
for key_idx, key in enumerate(classes_touch_paths):
    classes_touch_paths[key] = [item for sublist in classes_touch_paths[key] for item in sublist] # Flatten the list of lists into a single list
    all_labels += [cls_to_idx[key]] * len(classes_touch_paths[key])
print(f"Total touch samples: {len(all_labels)}")
touch_paths = [path for paths in classes_touch_paths.values() for path in paths] # Flatten the list of lists into a single list
touch_paths = list(zip(touch_paths, all_labels))
random.shuffle(touch_paths)
touch_paths, all_labels = zip(*touch_paths)

training_touch_paths = touch_paths[:int(0.1 * len(touch_paths))]
testing_touch_paths = touch_paths[int(0.1 * len(touch_paths)):]
trainin_label = all_labels[:int(0.1 * len(all_labels))]
testing_label = all_labels[int(0.1 * len(all_labels)):]
pd.DataFrame({"path": training_touch_paths, "label": trainin_label}).to_csv("YCB-Slide_touch_training_data.csv", index=False)
pd.DataFrame({"path": testing_touch_paths, "label": testing_label}).to_csv("YCB-Slide_touch_testing_data.csv", index=False)

# touch_data = data.load_and_transform_vision_data(touch_paths, device="cpu")
# touch_dataset = torch.utils.data.TensorDataset(touch_data, torch.tensor(all_labels))
# touch_training_dataset, touch_testing_dataset = torch.utils.data.random_split(touch_dataset, [int(0.1 * len(touch_dataset)), len(touch_dataset) - int(0.1 * len(touch_dataset))])

# torch.save(touch_training_dataset, "YCB-Slide_touch_training_dataset.pt")
# torch.save(touch_testing_dataset, "YCB-Slide_touch_testing_dataset.pt")