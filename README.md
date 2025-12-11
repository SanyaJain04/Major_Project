Hybrid Multimodal Content Moderation and Integrity Verification Framework Using Deep Learning and NLP
_____________________________________________________________________________________________________________________________________________________________________________________________________________________

A comprehensive AI-powered system that detects harmful online content across multiple modalities. 

The framework integrates:
Hate Speech Detection using transformer models (BERT, RoBERTa, HateBERT), logistic regression, CNN, SVM, LSTM, BiLSTM and hybrid models.
Morphed Image/Deepfake Detection using state-of-the-art vision models (ViT, EfficientNet-B3, SwinV2, ConvNeXt V2, Graph CNN, LSTM, GRU).

The system processes text, images, and video content to identify manipulated media and harmful language in real-time, providing automated moderation for online platforms.

<img width="349" height="660" alt="Project Architecture" src="https://github.com/user-attachments/assets/d5788d92-dfea-476e-8f4c-b906730617df" />

_____________________________________________________________________________________________________________________________________________________________________________________________________________________

Installation Instructions:
Quick Setup (Google Colab)
No installation needed! Simply follow these steps:

1) Open Google Colab:
Go to colab.research.google.com
Click on "GitHub" tab
Paste the repository URL.

2) Run Notebooks Directly:
Open any .ipynb file from the notebooks/ folder
Click "Runtime" → "Run all" or run each cell individually (▶️)
All dependencies install automatically
_____________________________________________________________________________________________________________________________________________________________________________________________________________________

Project Board
Current Status (Last Updated: December 2025)

Completed Tasks:
Research & Literature Survey - Comprehensive analysis of 20+ research papers
Dataset Collection - CASIA v2.0, Twitter Hate Speech

Model Development:
Text: LR, SVM, LSTM, BERT, RoBERTa, HateBERT, logistic regression, CNN, BiLSTM  and hybrid models implemented
Image: ViT, EfficientNet-B3, SwinV2, ConvNeXt V2, LSTM and GRU models trained
Evaluation & Testing - All models evaluated with standard metrics

In Progress:
Multimodal Fusion - Integrating text and image models
API Development - REST API for content moderation
Documentation - Finalizing technical documentation

Planned:
Real-time Deployment - Cloud deployment on AWS/GCP
Mobile Integration - Android/iOS SDK development
Cross-language Support - Multi-lingual hate speech detection

_____________________________________________________________________________________________________________________________________________________________________________________________________________________

