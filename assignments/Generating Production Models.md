# Generating Production Models

## Overview

Before you can begin Lab 3, your REST API needs models to serve. We have to run our training process, run many different algorithms and hyperparameter tune to produce good models.

In Lab 02, you learned how to use MLflow to track experiments. We will now run many different algorithms and with extensive hyperparameter configurations and track all of these using MLFlow. Then, we can identify top models, retrieve them from MLflow's artifacts, and save them in the correct location for Lab 3.

**Goal**: Train a variety of different models with extensive hyperparameter settings. Then we find your top two models from MLflow and save them to the `models/` directory.

---

## Step 0: ✅ Bug Fixes Already Applied in `src/train.py`

**Note**: The following bug fixes have already been applied to the codebase. This section is provided for reference to help you understand what was corrected.

### Bug Fix 1: Automatic Label Detection

The `src/train.py` file now automatically detects whether labels are numeric or string and sets the `pos_label` parameter accordingly. The code uses the following logic (around lines 568-577):

```python
# Determine the positive label automatically
unique_labels = np.unique(y_test)
if len(unique_labels) > 0:
    if np.issubdtype(unique_labels.dtype, np.number):
        pos_label = 1  # For numeric labels (from LabelEncoder)
    else:
        pos_label = sorted(unique_labels)[-1]  # For string labels
else:
    pos_label = 1  # Default fallback
```

This fix ensures compatibility with our `LabelEncoder`, which converts labels to numeric values (`0` and `1`).

### Bug Fix 2: Optional Model Saving

The `src/train.py` script now includes a `--no-save` / `--save` argument system to control whether models are saved locally. Since MLflow tracks all artifacts, local model saving is now **disabled by default**. The following has been added:

```python
save_group = parser.add_mutually_exclusive_group()
save_group.add_argument(
    '--no-save',
    action='store_true',
    dest='no_save',
    default=True,
    help='Do not save trained models to local disk (default)'
)
save_group.add_argument(
    '--save',
    action='store_false',
    dest='no_save',
    help='Save trained models to local disk'
)
```

The code now conditionally saves models based on this argument (around lines 850-890):

```python
model_path = None
if not args.no_save:
    model_path = save_model(model, model_name, args.output_dir)
    saved_paths[model_name] = model_path
models[model_name] = model

# Conditional logging
if model_path is not None:
    logger.info(f"Model saved: {model_path}")
    mlflow.log_artifact(model_path, "local_models")
else:
    logger.info("Model not saved to disk (default; use --save to enable)")
```

**Default behavior**: Models are **not saved** locally unless you use the `--save` flag. All model artifacts are tracked in MLflow.

## Step 1: Ensure Your Environment is Ready

1. Make sure your virtual environment is active.

    ```sh
    source .venv/bin/activate
    ```

2. Install all required dependencies, if there are changes.

    ```sh
    pip install -r requirements.txt
    ```

3. Pull your DVC-tracked data files, if you don't have them:

    ```sh
    dvc pull
    python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir data/processed --test-size 0.2
    ```

### 1.1: 🛑 **IMPORTANT**: Setting Up DVC Credentials

Before you can run `dvc pull`, you may get an `Unable to locate credentials` error. This is because DVC does not have the "password" to access your DagsHub remote storage.

This is a **one-time setup** you must do on any new environment (your local machine, a new CodeSpace instance, etc.).

#### A Quick Note: `dagshub login` vs. S3 Keys

You might be tempted to run the `dagshub login` command. This command is great, but it configures credentials for Git and the DagsHub API using a system called OAuth.

Our DVC remote, however, is configured to use the **S3 protocol** (as defined in `.dvc/config`). This protocol requires a different kind of "password": an **Access Key** and a **Secret Key**.

Therefore, `dagshub login` **will not fix** the DVC `Unable to locate credentials` error for this project. You *must* use the `dvc remote modify` commands shown below to set the S3 keys.

**1. Find Your Credentials on DagsHub**:

- Go to your DagsHub repository in your browser.
- Go to **Get started with Data** section towards the bottom of the page.
- Click on **Simple data upload** tab under **Configure your data storage**.
- On the bottom right hand, you will see **Connection credentials** box.
- Find your public key ID / secret access key (they have the same value) listed as **Public Key ID and Secret Access Key** (it may be hidden, you may have to click the eye icon to make it visible; there is also a copy button next to that that will copy that to your clipboard, so you can paste in your CLI command)

- **Alternatively**, you may see a blue **"Remote"** button on your DagsHub repository page. Click it and a window will pop up. You will see your **Access Key ID** and **Secret Access Key** which you can copy.

**2. Set Your Credentials in the Terminal**:

In your terminal, run these two commands. Replace the placeholders with the keys you just copied.

```sh
dvc remote modify origin --local access_key_id <YOUR_DVC_ACCESS_KEY_ID>
dvc remote modify origin --local secret_access_key <YOUR_DVC_SECRET_ACCESS_KEY>
```

---

## Step 2: Run Comprehensive Training & Tuning

Now we will run our full training and tuning pipeline from Lab 02. This will train all 6 of our models and use `GridSearchCV` to find the best hyperparameters for each, logging every single run to MLflow.

1. **Run Preprocessing (Safety Check)**:
    It's always good practice to ensure your processed data is up-to-date.

    ```sh
    python -m src.preprocess --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv --output-dir data/processed --test-size 0.2
    ```

2. **Run Training & Tuning**:

    This is the most important step. We run `src/train.py` and add the `--tune` flag. This will trigger `GridSearchCV` and may take several minutes to complete.

    ```sh
    python -m src.train --data data/processed/preprocessed_data.npy --model all --tune
    ```

---

## Step 3: Use MLflow to Find Your Best Models

After the script finishes, your `mlruns` directory is full of experiment data. We will use the MLflow UI to analyze it.

1. **Launch the MLflow UI**:
    In your terminal, run:

    ```sh
    mlflow ui --port 5000
    ```

    This will start the MLflow server. Open the URL it provides (usually `http://127.0.0.1:5000`) in your browser.

    > **Note for GitHub Codespaces**: Codespaces will automatically detect the running server and show a pop-up in the bottom-right corner asking if you want to open it in a browser. Click "Open in Browser."

2. **Analyze Experiments**:
    - In the MLflow UI, you will see your experiment (e.g., "Telecom Churn Prediction"). Click on it.
    - You will now see a table of all your runs.
    - Find the column for your main evaluation metric, such as `val_f1_score` or `val_accuracy`. Add the metric column you want if it is not there.
    - **Click the column header** (e.g., `val_f1_score`) to sort all your runs from highest to lowest.

3. **Identify Top 2 Models**:
    - The run at the very top is your **best model (v1)**.
    - The run just below it is your **second-best model (v2)**.

---

## Step 4: Save Your Models for Production

For each of your two top runs, you need to retrieve the saved model file.

1. **Create the `models/` Directory**:
    Our `app.py` looks for models in a `models/` folder. Create it in the root of your project if it doesn't exist.

    ```sh
    mkdir -p models
    ```

2. **Find the Artifact (Model v1)**:
    - Click on the name of your **best run** (e.g., "voting_classifier_tuned") in the MLflow UI.
    - On the run's page, scroll down to the "Artifacts" section.
    - You should see your saved model file (e.g., `model.pkl` or similar, it might be in a subfolder). Download this file.

3. **Save and Rename (Model v1)**:
    - Move the downloaded file into your new `models/` folder.
    - **Rename the file to `model_v1.pkl`**.

    ```sh
    # Example (after downloading):
    mv ~/Downloads/model.pkl models/model_v1.pkl
    ```

    *(Adjust the path from `~/Downloads/model.pkl` to wherever your browser saved the file.)*

4. **Save and Rename (Model v2)**:
    - Go back to the MLflow UI experiment list.
    - Click on your **second-best run**.
    - Repeat the process: find the model artifact, download it, and move it to the `models/` directory.
    - **Rename this second file to `model_v2.pkl`**.

---

## Step 5: Final Check

You are now ready for Lab 3! Your project must have the following files for the API to work:

- `models/model_v1.pkl`
- `models/model_v2.pkl`

---

## Congratulations! 🎊

**Model Generation Complete!**

You now have production models you can use for lab 3!

---

*Production Model Generation Instructions*  
*CMPT 2500: Machine Learning Deployment and Software Development*  
*NorQuest College*
