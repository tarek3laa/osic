# FVC
The project predicts values of forced vital capacity for pulmonary fibrosis patients. The dataset contains a mixture of structured data and image data, and I applied the model called a Least square dummy variable because the structured data is panel data and after training the model on the structured data I got a mean absolute error equals 150. Then I transformed the CT scan from axial view into coronal and sagittal view after that me and my friend created a data set of segmented images then we trained the u-net model using that data set and we got a binary cross-entropy loss of around 0.06.
after that, I extracted the lung features using the EfficientNetB0 model, then I trained the model using the structured data in addition to lung features.

for more details: https://prezi.com/view/wf7tfOVPIL0et1Ax7cZk/
