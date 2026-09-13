# Databricks notebook source
# MAGIC %pip install datasets==2.20.0 transformers==4.49.0 tf-keras==2.17.0 accelerate==1.4.0 mlflow==2.20.2 torchvision==0.20.1 deepspeed==0.14.4
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

training_df = spark.read.table("smart_claims_dev.02_silver.training_images")
display(training_df.limit(3))

# COMMAND ----------

display(training_df.columns)

# COMMAND ----------

import io

from pyspark.sql.functions import pandas_udf, col

IMAGE_RESIZE = 224

catalog = "smart_claims_dev"
silver_schema = "02_silver"

metadata_path = (
    "/Volumes/smart_claims_dev/00_landing/claims/autoloader_metadata"
)


@pandas_udf("binary")
def resize_image_udf(content_series):

    def resize_image(content):
        from PIL import Image

        image = Image.open(io.BytesIO(content))

        width, height = image.size
        new_size = min(width, height)

        # Center crop
        image = image.crop((
            (width - new_size) / 2,
            (height - new_size) / 2,
            (width + new_size) / 2,
            (height + new_size) / 2
        ))

        # Resize
        image = image.resize(
            (IMAGE_RESIZE, IMAGE_RESIZE),
            Image.NEAREST
        )

        # Convert to JPEG bytes
        output = io.BytesIO()
        image.save(output, format="JPEG")

        return output.getvalue()

    return content_series.apply(resize_image)


image_meta = {
    "spark.contentAnnotation": '{"mimeType": "image/jpeg"}'
}


# READ FROM SILVER AS STREAMING DATAFRAME
training_df = spark.readStream.table(
    "smart_claims_dev.02_silver.training_images"
)


# RESIZE AND WRITE
(
    training_df
    .withColumn(
        "content",
        resize_image_udf(col("content")).alias(
            "content",
            metadata=image_meta
        )
    )
    .writeStream
    .option(
        "checkpointLocation",
        f"{metadata_path}/_checkpoint_resized_v2"
    )
    .trigger(availableNow=True)
    .toTable(
        "smart_claims_dev.02_silver.training_images_resized"
    )
)

# COMMAND ----------


display(spark.table("smart_claims_dev.02_silver.training_images_resized").limit(10))

# COMMAND ----------

from datasets import Dataset
import mlflow

#Setup the training experiment
mlflow.set_experiment(
    "/Users/sahilmohril@gmail.com/smart_claims_image_classification"
)
df = spark.table(
    "smart_claims_dev.02_silver.training_images_resized"
)

# Convert to Pandas
pdf = df.toPandas()

# Convert Pandas → Hugging Face Dataset
dataset = Dataset.from_pandas(
    pdf,
    preserve_index=False
)

# Rename image column
dataset = dataset.rename_column(
    "content",
    "image"
)

# Train/validation split
splits = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_ds = splits["train"]
val_ds = splits["test"]

# COMMAND ----------

import torch
from transformers import AutoFeatureExtractor, AutoImageProcessor

# pre-trained model from which to fine-tune
# Check the hugging face repo for more details & models: https://huggingface.co/microsoft/resnet-50
model_checkpoint = "microsoft/resnet-50"

from PIL import Image
import io
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomResizedCrop, Resize, ToTensor, Lambda

#Extract the model feature (contains info on pre-process step required to transform our data, such as resizing & normalization)
#Using the model parameters makes it easy to switch to another model without any change, even if the input size is different.
model_def = AutoFeatureExtractor.from_pretrained(model_checkpoint)

#Transformations on our training dataset. we'll add some crop here
transforms = Compose([Lambda(lambda b: Image.open(io.BytesIO(b)).convert("RGB")), #byte to pil
                        ToTensor(), #convert the PIL img to a tensor
                        Normalize(mean=model_def.image_mean, std=model_def.image_std)
                        ])

# Add some random resiz & transformation to our training dataset
def preprocess(batch):
    """Apply train_transforms across a batch."""
    batch["image"] = [transforms(image) for image in batch["image"]]
    return batch
   
#Set our training / validation transformations
train_ds.set_transform(preprocess)
val_ds.set_transform(preprocess)

# COMMAND ----------


from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

#Mapping between class label and value (huggingface use it during inference to output the proper label)
label2id, id2label = dict(), dict()
for i, label in enumerate(set(dataset['label'])):
    label2id[label] = i
    id2label[i] = label
    
#Load the base model from its checkpoint
model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint, 
    label2id=label2id,
    id2label=id2label,
    num_labels=len(label2id),
    ignore_mismatched_sizes = True # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
     

# COMMAND ----------


model_name = model_checkpoint.split("/")[-1]

from transformers import TrainingArguments
args = TrainingArguments(
    f"/tmp/huggingface/pcb/{model_name}-finetuned",
    no_cuda=True, #Run on CPU for resnet to make it easier
    remove_unused_columns=False,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    num_train_epochs=20,
    load_best_model_at_end=True
)

# COMMAND ----------

import mlflow
# This wrapper adds steps before and after the inference to simplify the model usage
# Before calling the model: apply the same transform as the training, resizing the image
# After callint the model: only keeps the main class with the probability as output
class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, pipeline):
        self.pipeline = pipeline
        # instantiate model in evaluation mode
        self.pipeline.model.eval()

    def predict(self, context, images):
        from PIL import Image
        with torch.set_grad_enabled(False):
            #Convert the byte to PIL images
            images = images['content'].apply(lambda b: Image.open(io.BytesIO(b))).to_list()
            #the pipeline returns the probability for all the class
            predictions = self.pipeline.predict(images)
            #Filter & returns only the class with the highest score [{'score': 0.999038815498352, 'label': 'normal'}, ...]
            return pd.DataFrame([max(r, key=lambda x: x['score']) for r in predictions])

# COMMAND ----------

from transformers import pipeline, DefaultDataCollator, EarlyStoppingCallback
from mlflow.models import infer_signature

with mlflow.start_run(run_name="hugging_face_new") as run:
    mlflow.log_input(mlflow.data.from_huggingface(train_ds, "training"))

    # use real class count instead of 3
    def collate_fn(examples):
        import torch
        pixel_values = torch.stack([e["image"] for e in examples])
        labels = torch.tensor([label2id[e["label"]] for e in examples], dtype=torch.long)
        labels = torch.nn.functional.one_hot(labels, num_classes=len(label2id)).float()
        return {"pixel_values": pixel_values, "labels": labels}

    trainer = Trainer(model, args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=model_def, data_collator=collate_fn)
    train_results = trainer.train()

    # Build final HF pipeline
    classifier = pipeline("image-classification", model=trainer.state.best_model_checkpoint, tokenizer=model_def)

    # ---- moved from your Cell B, so it's inside the SAME run ----
    import pandas as pd
    wrapped_model = ModelWrapper(classifier)
    test_df = spark.table("smart_claims_dev.02_silver.training_images_resized").select('content').toPandas()
    predictions = wrapped_model.predict(None, test_df)
    signature = infer_signature(test_df, predictions)

    reqs = mlflow.transformers.get_default_pip_requirements(model)
    
    # LOG the model and CAPTURE the URI
    logged = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        pip_requirements=reqs,
        signature=signature,
    )

# keep these prints to sanity-check
from mlflow import artifacts
print("logged.model_uri:", logged.model_uri)   # e.g., runs://model
print("logged.run_id  :", logged.run_id)
print("model files    :", artifacts.list_artifacts(logged.model_uri))

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
model_name = "smart_claims_dev.03_gold.claims_damage_level"

registered = mlflow.register_model(
    model_uri=logged.model_uri,
    name=model_name,
)

MlflowClient().set_registered_model_alias(
    name=model_name,
    alias="prod",
    version=registered.version,
)

print(f"Registered {model_name} v{registered.version} and set alias 'prod'.")

# COMMAND ----------


predict_damage_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/smart_claims_dev.03_gold.claims_damage_level@prod")
columns = predict_damage_udf.metadata.get_input_schema().input_names()
#Run the inferences
spark.table('smart_claims_dev.02_silver.training_images_resized').withColumn("damage_prediction", predict_damage_udf(*columns)).write.mode('overwrite').saveAsTable('smart_claims_dev.03_gold.damage_predictions')

# COMMAND ----------


predictions = spark.table('smart_claims_dev.03_gold.damage_predictions')
display(predictions)

# COMMAND ----------

results = predictions.selectExpr("path", "label", "damage_prediction.label as predictions", "damage_prediction.score as score").toPandas()

# COMMAND ----------


import matplotlib.pyplot as plt
import seaborn as sns

# create confusion matrix
confusion_matrix = pd.crosstab(results['label'], results['predictions'])

# plot confusion matrix
fig = plt.figure()
sns.heatmap(confusion_matrix, annot=True, cmap="Blues", fmt='d')

# COMMAND ----------

from pyspark.sql.functions import col

raw_images = (
    spark.read.table("smart_claims_dev.02_silver.claim_images")
    .withColumnRenamed("image_content", "content")
    .withColumn(
        "damage_prediction",
        predict_damage_udf(*columns)
    )
)

display(raw_images)

# COMMAND ----------

metadata = (
    spark.table("smart_claims_dev.01_bronze.claim_images_metadata")
    .select(
        "image_name",
        "image_id",
        "claim_no",
        "chassis_no"
    )
)



# COMMAND ----------

from pyspark.sql.functions import broadcast

# Keep only columns needed for the Gold table
predictions = raw_images.select(
    "image_name",
    "image_path",
    "damage_prediction"
)

# Keep only required metadata columns
metadata_small = metadata.select(
    "image_name",
    "image_id",
    "claim_no",
    "chassis_no"
)

# If metadata is relatively small, broadcast it
result = predictions.join(
    broadcast(metadata_small),
    on="image_name",
    how="left"
)

result.write \
    .mode("overwrite") \
    .saveAsTable("smart_claims_dev.03_gold.claim_images_predicted")