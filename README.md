![image](https://github.com/user-attachments/assets/5667a6b7-3b92-444e-bad9-1c87310d3b4e)
This is a multimodal smart contract vulnerability detection method based on federated and comparative learning, and the method model is shown in the figure above. 

The model is divided into two parts: global model and local model. global model trains the data of all modalities and obtains the corresponding embedding representations. 
Local model is R_global^text and R_global^image, respectively: R_global^text and R_global^image, respectively. local model is divided into text and image models, 
and obtains the embedding representations R_local^text and R_local^image for text and image, respectively. The local model is divided into text and image models, 
and obtains the embedding representations of text R_local^text and image R_local^image, respectively, and then uploads them to the global model.

The global model implements contrast learning: R_global^text-R_local^image and R_global^image-R_local^image, which allows the local unimodal model to move towards 
the shared multimodal representation space while staying away from other irrelevant text data points. The results of the aggregated comparative learning are then 
federated distillation to update the learned parameters back to the local model to avoid re-training and improve model efficiency.
