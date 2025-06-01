import yaml
import mlflow
import os
import random
import glob
from src import data_preprocessing, train, evaluate, predict, speech

def run_pipeline():
    with open('src/config.yaml') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])

    print("Loading data...")
    train_gen, test_gen = data_preprocessing.load_and_preprocess_data(config)

    with mlflow.start_run():
        print("Building model...")
        model = train.build_model(config)

        print("Training model...")
        history = train.train_model(model, train_gen, test_gen, config)

        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", config['model']['name'])
        model.save(model_path)

        mlflow.log_param("epochs", config['model']['epochs'])
        mlflow.log_param("batch_size", config['dataset']['batch_size'])
        mlflow.log_metric("val_accuracy", history.history['val_accuracy'][-1])
        mlflow.log_artifact(model_path)

    print("Evaluating model...")
    evaluate.evaluate_model(model, test_gen)

    test_dir = config['dataset']['test_path']
    class_dirs = [os.path.join(test_dir, d) for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    sample_class_dir = random.choice(class_dirs)
    sample_image = random.choice(glob.glob(os.path.join(sample_class_dir, "*")))

    print(f"Predicting on sample image: {sample_image}")
    pred = predict.predict(model, sample_image, config)
    print(f"Prediction: {pred}")

    speech.speak_text(pred)

    print("Pipeline finished.")

if __name__ == "__main__":
    run_pipeline()

