AgeLens AI — Deep Learning Age Estimation

Estimate apparent age from facial images using a fine-tuned ResNet18 convolutional neural network built with PyTorch.

AgeLens AI is an end-to-end computer vision project covering image preprocessing, CNN-based regression, transfer learning, model evaluation, and cloud deployment.
<img width="1912" height="921" alt="image" src="https://github.com/user-attachments/assets/dcc62769-ab56-4a1a-8a9d-5e86652a8e3a" />



🚀 Live Demo

Try AgeLens AI: https://agelensai.streamlit.app/

Upload a facial photograph and receive an AI-generated apparent-age estimate.

✨ Features

Image-based age estimation: Predict a continuous age value rather than classifying images into age categories.

Transfer learning: Fine-tuned a pretrained ResNet18 model for an image-regression task.

Image preprocessing: RGB conversion, resizing to 224 × 224 pixels, tensor conversion, and ImageNet normalization.

Model hosting: Store model weights in a private Hugging Face repository instead of bundling the checkpoint with the application code.

Interactive web interface: Upload images and view predictions through a custom dark-themed Streamlit UI.

Cloud deployment: Deployed using Streamlit Community Cloud.

Inference pipeline: Load the trained checkpoint, prepare the image, run inference, and display the predicted value.

🧠 Model Architecture

AgeLens AI uses ResNet18, a convolutional neural network originally pretrained on ImageNet.

The original classification head is replaced with a single-output linear layer to perform regression.

Inference pipeline

Input: Upload a facial image.

Preprocessing: Convert to RGB, resize to 224 × 224, convert to a tensor, and apply ImageNet normalization.

Feature extraction: Process the image through the ResNet18 backbone.

Regression: Predict a continuous numerical age estimate.

Output: Display the estimated age in years.

📊 Model Evaluation

The fine-tuned model achieved the following validation result:

Metric

Result

Validation MAE

6.14 years

Dataset

UTKFace

Architecture

Fine-tuned ResNet18

Task

Image regression

What does MAE mean?

Mean Absolute Error measures the average absolute difference between predicted and actual ages. A validation MAE of 6.14 years means the average absolute prediction error on the validation set was approximately 6.14 years.

This is a validation result, not a guarantee of accuracy on unseen real-world photographs. No separate test-set result is reported here.

Validation error by age group

Age group

MAE

0–17

4.34 years

18–29

4.41 years

30–44

6.09 years

45–59

8.42 years

60+

11.21 years

The observed error increased across older age groups in this validation split. Model predictions can vary with age, image quality, lighting, and demographic representation.

🛠️ Tech Stack

Language: Python

Deep learning: PyTorch

Computer vision: Torchvision, Pillow

Web application: Streamlit

Model hosting: Hugging Face Hub

Deployment: Streamlit Community Cloud

Dataset: UTKFace

📁 Project Structure

AgeLens-AI/
├── app.py
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml
└── screenshots/
    └── agelens-demo.png


🔐 Model Hosting and Deployment

The trained model weights are hosted separately in a private Hugging Face repository.

The application uses hf_hub_download to retrieve the checkpoint and loads the weights into the matching ResNet18 architecture.

For deployment:

Push the application code and requirements.txt to GitHub.

Configure HF_TOKEN in Streamlit Community Cloud's Secrets settings.

Deploy the repository through Streamlit Community Cloud.

Upload a test image and verify that inference completes successfully.

The Hugging Face token must be configured as a deployment secret, not hardcoded in source code.

⚠️ Limitations and Responsible Use

Predictions are estimates and may be inaccurate for individual photographs.

Performance differs across age groups and image conditions.

The validation MAE does not establish real-world accuracy for every population.

The model should not be used for identity verification, medical decisions, eligibility decisions, or other high-stakes applications.

Only upload photographs when you have appropriate permission to use them.

🎯 Learning Outcomes

This project provided hands-on experience with:

CNN-based image regression.

Transfer learning and fine-tuning pretrained models.

Image normalization and inference preprocessing.

Loading and serving PyTorch model checkpoints.

Integrating Hugging Face Hub with a web application.

Managing secrets and dependencies in a cloud environment.

Deploying an end-to-end computer vision application.

👨‍💻 Author

Yash Bhatt

B.Tech — Artificial Intelligence and Machine Learning

Interested in Machine Learning, Deep Learning, Computer Vision, and building end-to-end AI applications.

AgeLens AI is an educational machine learning project built to explore practical computer vision and model deployment.
