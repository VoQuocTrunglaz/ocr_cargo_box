# OCR API Deployment to EC2 with GitHub Actions

This project sets up a CI/CD pipeline to deploy an OCR (Optical Character Recognition) API to an AWS EC2 instance using Docker and GitHub Actions. The API leverages YOLO for object detection, PaddleOCR for text recognition, and the Gemini API for text classification to extract and process data from cargo box images. The application is containerized, pushed to GitHub Container Registry (GHCR), and deployed to an EC2 instance running Ubuntu.

## 📋 Overview

- **Application**: A Python-based OCR API for extracting and classifying text from cargo box images
Technologies:
- YOLO: Detects text regions in images.
- PaddleOCR: Recognizes text within detected regions.
- Gemini API: Classifies and structures extracted text into JSON.
- **CI/CD Pipeline**: Automated via GitHub Actions for testing, building, and deployment.
- **Deployment Target**: AWS EC2 instance running Ubuntu.
- **Container Registry**: GitHub Container Registry (GHCR).

The pipeline triggers on every push to the `main` branch. It runs tests, builds a Docker image, pushes it to GHCR, and deploys it to an EC2 instance.

## 📦 Prerequisites

Before setting up the project, ensure you have the following:

- **AWS EC2 Instance**:
  - An Ubuntu-based EC2 instance with port 80 open (for HTTP access).
  - SSH access enabled (port 22).
  - Security group configured to allow inbound traffic on ports 80 and 22.

- **GitHub Secrets**:
  - `AWS_HOST`: The public IP or DNS of your EC2 instance.
  - `AWS_KEY`: The private SSH key (in PEM format) to access your EC2 instance.
  - `GHCR_PAT`: A GitHub Personal Access Token (PAT) with `read:packages` and `write:packages` scopes to access GHCR.

## 🛠️ Setup Instructions

1. **Clone the Repository**:
   Clone this repository to your local machine:
   ```
   git clone https://github.com/VoQuocTrunglaz/ocr_cargo_box_api.git
   cd ocr_cargo_box_api
   ```

2. **Configure GitHub Secrets**:
   - Go to your GitHub repository > **Settings** > **Secrets and variables** > **Actions**.
   - Add the following secrets:
     - `AWS_HOST`: Your EC2 instance's public IP or DNS.
     - `AWS_KEY`: Your EC2 private SSH key (copy the entire PEM file content).
     - `GHCR_PAT`: Your GitHub PAT with package read/write permissions.

3. **Set Up the EC2 Instance**:
   - **Launch an EC2 Instance**:
     Launch an Ubuntu EC2 instance on AWS.
   - **Create a Key Pair (if you don’t have one)**:
     To access the EC2 instance via SSH, you need a key pair. If you don’t already have one, create it as follows:
     1. Go to the AWS Management Console > EC2 > **Key Pairs** (under **Network & Security**).
     2. Click **Create key pair**.
     3. Name your key pair (e.g., `my-ec2-key`), select the file format as `.pem`, and click **Create key pair**.
     4. The `.pem` file (e.g., `my-ec2-key.pem`) will be downloaded to your computer. Store it in a secure location.
     5. Change the file permissions to secure it (on Linux/Mac):
        ```
        chmod 600 my-ec2-key.pem
        ```
     - When launching the EC2 instance, select this key pair in the **Key pair (login)** step.
   - **Configure the Security Group**:
     Configure the security group to allow inbound traffic on:
     - Port 22 (SSH) for GitHub Actions to connect.
       - **Type**: SSH
       - **Protocol**: TCP
       - **Port range**: 22
       - **Source**: `0.0.0.0/0` (or restrict to your IP for added security).
     - Port 80 (HTTP) to access the OCR API.
       - **Type**: HTTP
       - **Protocol**: TCP
       - **Port range**: 80
       - **Source**: `0.0.0.0/0`.
     - Port 8000 (Custom TCP) for direct access to the OCR API (if not mapping to port 80).
       - **Type**: Custom TCP
       - **Protocol**: TCP
       - **Port range**: 8000
       - **Source**: `0.0.0.0/0` (or restrict to your IP for added security).
       - **Description (optional)**: "Allow OCR API access on port 8000".
     - To add these rules:
       1. Go to AWS Management Console > EC2 > Security Groups.
       2. Select the security group associated with your EC2 instance.
       3. Go to the **Inbound rules** tab > Click **Edit inbound rules**.
       4. Add the rules as described above > Click **Save rules**.
   - **Ensure the EC2 Instance Has a Key Pair**:
     When launching the EC2 instance, in the **Key pair (login)** step, select the key pair you created (or an existing one). The private key of this key pair will be used as the `AWS_KEY` in GitHub Secrets.

4. **Add the GitHub Actions Workflow**:
   - The `.github/workflows/ci-cd.yml` file in this repository defines the CI/CD pipeline.
   - It includes three jobs: `pre_build` (setup and testing), `build` (Docker image build and push), and `deploy` (deploy to EC2).
   - Ensure this file is in your repository’s `.github/workflows/` directory.

5. **Push to Trigger Deployment**:
   - Commit and push your changes to the `main` branch:
     ```
     git add .
     git commit -m "Initial commit"
     git push origin main
     ```
   - This will trigger the GitHub Actions workflow.

## 🚀 Usage

- **Access the API**:
  After the workflow completes, your OCR API will be running on the EC2 instance.
  - Open a browser or use a tool like `curl` to access the API at `http://<EC2_PUBLIC_IP>` (e.g., `http://ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com`).
  - If you’re not mapping port 80, you can access the API directly on port 8000: `http://<EC2_PUBLIC_IP>:8000`.
  - Example request:
    ```
    curl -X POST -F "image=@/path/to/cargo_box_image.jpg" http://<EC2_PUBLIC_IP>/ocr
    ```

- **Monitor the Workflow**:
  - Go to your GitHub repository > **Actions** tab to view the workflow logs.
  - Check for any errors during the pre-build, build, or deployment steps.

## ⚠️ Troubleshooting

- **SSH Connection Issues**:
  - Ensure the `AWS_HOST` and `AWS_KEY` secrets are correct.
  - Verify that port 22 is open on your EC2 instance’s security group.
  - Check the SSH key permissions (`chmod 600` on the key file).

- **Docker Image Push Fails**:
  - Verify that the `GHCR_PAT` has the correct permissions.
  - Ensure you’re logged in to GHCR correctly in the workflow.

- **API Not Accessible**:
  - Confirm that port 80 (or port 8000 if not mapped) is open on your EC2 instance.
  - Check the container logs on EC2:
    ```
    ssh -i <your-key.pem> ubuntu@<EC2_PUBLIC_IP>
    docker logs ocr_api
    ```

- **Tests Fail**:
  - Ensure all dependencies are listed in `requirements.txt`.
  - Check the test logs in the GitHub Actions workflow for details.

## 📝 Notes

- The OCR API runs on port 8000 inside the container, but it’s mapped to port 80 on the EC2 instance (`-p 80:8000`). If you want to access it directly on port 8000, ensure the port is open as described above.
- The container is set to restart automatically (`--restart=always`) to ensure it runs even after an EC2 reboot.
