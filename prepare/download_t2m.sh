mkdir -p checkpoints
cd checkpoints

echo -e "Downloading pretrained models for SALAD on the HumanML3D dataset"
gdown --folder https://drive.google.com/drive/folders/1YuDQCgc6RJ4WlR9vt_L34nkXChC7C98_?usp=drive_link

cd ..

echo -e "Downloading done!"