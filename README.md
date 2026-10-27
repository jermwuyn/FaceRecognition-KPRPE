# Face Recognition with Vit & KPRPE

This is a personal research project where I aim to build a reliable face recognition model using an architecture I found interesting through my research and experiments. I hope to use this project as a foundation for my future computer vision applications.

Status: Currently working on add more datasets (MS1MS2) for better training result... Demo coming soon.. 

The architecture in this model is : Vision Tranformer (ViT) + KP-RPE + ArcFace.
  + ViT: Self-attention helps the model learn relationships between different parts of the face like a metric and capture all information from the whole image
  + KP-RPE: Adds geometric information to attention mechanism by using the relative positions of five facial landmarks (Which detected by YuNet)
  + ArcFace: Encourages the model to produce similar embeddings for the same person and more distinguishable embedding for different people

## Why did i choose this architecture?
I previously experimented with ResNet101 + ArcFace for face recognition. ResNet101 uses Convolutional Neural Networks (CNNs) to extract facial features. It starts by learning local details, such as edges, eyes, and noses, and gradually combines them into a complete face representation. This approach is effective, but recognition performance may decrease when images are blurry, faces are viewed from different angles, or face alignment is inaccurate.

In contrast, Vision Transformer (ViT) divides an image into smaller patches and uses Self-Attention to learn relationships between different facial regions. Instead of mainly combining information through convolutional layers, ViT can directly capture connections between distant parts of the face.

However, a standard ViT does not directly use the geometric structure of a face. Therefore, I decided to add KeyPoint Relative Position Encoding (KP-RPE), which uses five facial landmarks—the eyes, nose, and mouth corners—to add geometric information to the attention mechanism.

This is why I chose ViT + KP-RPE + ArcFace as the next step in my research, rather than continuing with a traditional CNN-based architecture.

## How everything works
1. Face Preprocessing: YuNet detects five facial landmarks (eyes, nose, mouth) from 112x112 face image
2. ViT+KPRPE: ViT divides the image into 196 patches while KP-RPE add informations about facial geometry
3. Feature Extraction: Generates 256-dimensional face embedding presenting facial features
4. Training: ArcFace helps model learn to distinguish different identities by their features
5. Recognition: Cosine similarity compares two face embeddings to determine whether they are the same person

## Project Status:

  + Model: ViT + KP-RPE + ArcFace implemented. 
  + LFW: Completed 30 training epochs, with a best validation ROC-AUC of 0.6971 on a custom identity-disjoint split.
  + MS1MV2: Prepared a candidate subset of 100,000 images from 2,000 identities.
  + Next Steps: Complete MS1MV2 preprocessing, continue training experiments, and prepare for a demo.
