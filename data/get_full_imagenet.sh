mkdir imagenet_full
cd imagenet_full
wget -O list_of_images.txt "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n02139199"
wget --timeout 1 --tries 2 -i list_of_images.txt
