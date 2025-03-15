#!binbash
echo Unlocking encrypted dataset...
sudo cryptsetup luksOpen devsdb nudeimages
sudo mount devmappernudeimages ~nude_detection_projectdata

echo Starting training...
docker-compose up --build

echo Training complete. Locking dataset...
sudo umount ~nude_detection_projectdata
sudo cryptsetup luksClose nudeimages
