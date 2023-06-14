We provide codes for two main computer visions: segmentation and detection.

Folders:
---
- Docker: A docker container which is possible to run the notebooks
- images: graphical representation of the fusions
- Utilities:
- - configuration: a dictionary that contains hyper-parameters of each CNN
- - FusionWithTL: A class that generate the three levels of fusion with VGG16 (transfer learning)
- - FusionWithoutTL: A class that generate the three levels of fusion
- - utilities: A class that contains utilities such as load images, output generation etc. 
---
Files:
---
- Early_segmenation_fusion.ipynb a notebook with segmentation early fusion and transfer learning
- Early_segmentation_fusion_no_tl.ipynb the same early fusion but without transfer learning
- Middle_segmentation_fusion.ipynb a notebook with segmentation using middle fusion and transfer learning 
- Middle_segmentation_fusion_no_tl.ipynb a notebook with segmentation using middle fusion without transfer learning
- Late_segmentation_fusion.ipynb a notebook with segmentation with late fusion and transfer learning 
- Late_segmentation_fusion_no_tl.ipynb a notebook with segmentation with late fusion
---


