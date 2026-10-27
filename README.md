# Face Recognition with Vit & KPRPE

This is a personal research project with the idea of building the high quality face recognition model with the best architecure i've found and experiments with. I've try this to be the base for my future vision applications. 

The architecture in this model is : Vision Tranformer (ViT) + KP-RPE with the use of ArcFace.
  + ViT: Self-attention helps the model learn relationships between different parts of the face like a metric and capture all information from the whole image
  + KP-RPE: Add geometric information to attention mechanism by using the relative positions of five facial landmarks (Which detected by YuNet)
  + ArcFace: Encourages the model to produce similar embeddings for the same person and more distinguishable embedding for different people

