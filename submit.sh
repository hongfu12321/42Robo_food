
BUCKET=$GCS_BUCKET

TRAINER_PACKAGE_PATH="./trainer"
MAIN_TRAINER_MODULE="trainer.task"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="pizza_agent_$now"

JOB_DIR=$BUCKET$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --region us-central1 \
    --config config.yaml \
    --runtime-version 1.10 \
    --python-version 3.5 \
    -- \
    --rows 12 \
    --columns 12 \
    --n-epoch 5000 \
    --n-episodes 200 \
    --max-steps 1440 \
    --learning-rate 0.0005 \
    --output-dir $BUCKET"pizza_agent_$now" \
    --gamma 0.95 \
    --min-ingred 1 \
    --max-ingred 6 \