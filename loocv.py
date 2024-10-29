import os
import shutil
from ultralytics import YOLO
from roboflow import Roboflow

rf = Roboflow(api_key="Cas8UHCEClVzq9fqEFFu")
project = rf.workspace("eshita").project("apneaeye")
version = project.version(2)
dataset = version.download("yolov8")

base_path = dataset.location  
train_images_dir = os.path.join(base_path, 'train/images')
train_labels_dir = os.path.join(base_path, 'train/labels')
test_images_dir = os.path.join(base_path, 'test/images')
test_labels_dir = os.path.join(base_path, 'test/labels')
output_dir = os.path.join(base_path, 'folds')  # Folder to save folds
os.makedirs(output_dir, exist_ok=True)

train_image_files = os.listdir(train_images_dir)
test_image_files = os.listdir(test_images_dir)

all_image_files = train_image_files + test_image_files

# extract unique patient IDs
patient_ids = sorted(set([filename.split('_')[1] for filename in all_image_files]))  

# model = YOLO('yolov8n.pt') 
# model.train(data='/home/eshita.khanna/ApneaEye-2/data.yaml', epochs=100, batch=32, imgsz=640, val=False, device=0) 
# results = model.val(data='/home/eshita.khanna/ApneaEye-2/data.yaml', device=0)
# print("Without LOOCV Results:")
# print(results)

# to create the train/test split for a given patient
def create_fold_for_patient(patient_id):
    fold_dir = os.path.join(output_dir, f'fold_AP_{patient_id}')
    train_image_dir = os.path.join(fold_dir, 'train/images')
    test_image_dir = os.path.join(fold_dir, 'test/images')
    train_label_dir = os.path.join(fold_dir, 'train/labels')
    test_label_dir = os.path.join(fold_dir, 'test/labels')
    
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Separate images and labels into train and test based on patient ID
    for filename in all_image_files:
        # check if the current file belongs to the patient
        if f'AP_{patient_id}' in filename:
            # move patient-specific images to the test folder
            if filename in train_image_files:
                shutil.copy(os.path.join(train_images_dir, filename), os.path.join(test_image_dir, filename))
                label_filename = filename.replace('.jpg', '.txt')
                shutil.copy(os.path.join(train_labels_dir, label_filename), os.path.join(test_label_dir, label_filename))
            elif filename in test_image_files:
                shutil.copy(os.path.join(test_images_dir, filename), os.path.join(test_image_dir, filename))
                label_filename = filename.replace('.jpg', '.txt')
                shutil.copy(os.path.join(test_labels_dir, label_filename), os.path.join(test_label_dir, label_filename))
        else:
            # move other patient images to the train folder
            if filename in train_image_files:
                shutil.copy(os.path.join(train_images_dir, filename), os.path.join(train_image_dir, filename))
                label_filename = filename.replace('.jpg', '.txt')
                shutil.copy(os.path.join(train_labels_dir, label_filename), os.path.join(train_label_dir, label_filename))
            elif filename in test_image_files:
                shutil.copy(os.path.join(test_images_dir, filename), os.path.join(train_image_dir, filename))
                label_filename = filename.replace('.jpg', '.txt')
                shutil.copy(os.path.join(test_labels_dir, label_filename), os.path.join(train_label_dir, label_filename))

# patient_id = '20'
# print(f"Creating fold for patient: AP_{patient_id}")
# create_fold_for_patient(patient_id)
# print("Leave-one-out cross-validation setup complete!")

base_path_1 = '/home/eshita.khanna/ApneaEye-2/folds'  
output_base_dir = '/home/eshita.khanna/runs4'

def run_loocv(patient_id):
    # Define the paths for the current fold
    fold_dir = os.path.join(base_path_1, f'fold_AP_{patient_id}')
    train_data_path = os.path.join(fold_dir, 'train/images')
    train_labels_path = os.path.join(fold_dir, 'train/labels')
    test_data_path = os.path.join(fold_dir, 'test/images')
    test_labels_path = os.path.join(fold_dir, 'test/labels')

    data_config = {
        'train': train_data_path,
        'val': test_data_path,
        'nc': 7, 
        'names': ['blanket_overhead', 'face', 'mask', 'nostril', 'person_left', 'person_right', 'person_supine']
    }

    # Save the data configuration to a YAML file
    data_config_path = os.path.join(fold_dir, 'data.yaml')
    with open(data_config_path, 'w') as f:
        f.write('train: ' + data_config['train'] + '\n')
        f.write('val: ' + data_config['val'] + '\n')
        f.write('nc: ' + str(data_config['nc']) + '\n')
        f.write('names: ' + str(data_config['names']) + '\n')


    model = YOLO('yolov8m.pt') 
    model.train(data=data_config_path, epochs=100, batch=16, imgsz=640, val=False, device=1, project=output_base_dir, name=f'AP_{patient_id}') 
    results = model.val(data=data_config_path)

    print(f"Results for patient: AP_{patient_id}")
    print(results)

# p_id = '02'  
# run_loocv(p_id)

for patient_id in patient_ids[15:17]:
    print(f"Running LOOCV for patient: AP_{patient_id}")
    run_loocv(patient_id)

# model = YOLO('runs4/AP_20/weights/best.pt')
# results = model.val()
# print(results)
