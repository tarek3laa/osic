# Predicting Forced Vital Capacity (FVC) for Pulmonary Fibrosis Patients

This project aims to predict Forced Vital Capacity (FVC) for patients with pulmonary fibrosis, utilizing a combination of structured data analysis, image data segmentation, and deep learning techniques. 

## Structured Data Analysis

For the structured data component, we used the Least Square Dummy Variable model to handle the structured panel data. The model achieved a mean absolute error (MAE) of approximately 150, demonstrating the successful modeling of structured data.

## Image Data Processing

The project involved transforming CT scan images from axial view into coronal and sagittal views. Subsequently, we created a dataset of segmented lung images using the U-Net model for lung segmentation. The binary cross-entropy loss for the segmentation task was around 0.06, indicating the effectiveness of the U-Net model for image segmentation.

## Feature Extraction

We extracted lung features from the segmented images using the EfficientNetB0 model, enhancing our understanding of the patients' lung conditions.

## Integrated Modeling

The final predictive model was trained using the combined structured data and lung features. This comprehensive approach, which integrates structured data analysis and image-based deep learning, holds significant potential for improving the diagnosis and prognosis of pulmonary fibrosis patients.

For more details and a deeper dive into the project methodology and results, please refer to our [Prezi presentation](https://prezi.com/view/wf7tfOVPIL0et1Ax7cZk/).

This project showcases a powerful blend of structured data analysis and image-based deep learning, offering a holistic solution for predicting FVC in pulmonary fibrosis patients.
