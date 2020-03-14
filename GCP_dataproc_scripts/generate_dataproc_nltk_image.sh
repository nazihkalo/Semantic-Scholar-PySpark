## run this first gcloud config set dataproc/region us-central1

python generate_custom_image.py \
    --image-name nltk-image-bigdata-2 \
    --dataproc-version 1.4.2-debian9 \
    --customization-script customization-script.sh \
    --zone us-central1-a \
    --gcs-bucket gs://semantic_scholar_files