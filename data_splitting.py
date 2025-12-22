import os
import shutil
import random

# Caminhos (ajusta conforme o teu PC)
source_folder = 'celeba_hq_256'
base_data = 'data'

# Estrutura sugerida
folders = [
    '01_Real/train', 
    '01_Real/val', 
    '02_Fake_Gen', 
    '03_Test_Pairs/Real', 
    '03_Test_Pairs/Fake'
]

for f in folders:
    os.makedirs(os.path.join(base_data, f), exist_ok=True)

# Obter lista de imagens e baralhar
images = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png'))]
random.shuffle(images)

# Divisão (30k imagens)
train_imgs = images[:25000]
val_imgs = images[25000:27500]
test_imgs = images[27500:]

def move_files(files, dest):
    for f in files:
        shutil.move(os.path.join(source_folder, f), os.path.join(base_data, dest, f))

move_files(train_imgs, '01_Real/train')
move_files(val_imgs, '01_Real/val')
move_files(test_imgs, '03_Test_Pairs/Real')

