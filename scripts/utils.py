import torch
from PIL import Image
from io import BytesIO
import pickle
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

np.long = np.int_
np.ulong = np.uint

if not hasattr(np, "bool"):
    np.bool = bool
    
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

ROOT_PATH = "data/faces_webface_112x112"

class PairDataset(Dataset):
    def __init__(self, dataset_path, images_are_bytes = False):
        dataset_path = ROOT_PATH + "/" + dataset_path
        with open(dataset_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
            self.images = data[0]
            self.labels = data[1]

        self.images_are_bytes = images_are_bytes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image1_data = self.images[index * 2]
        image2_data = self.images[index * 2 + 1]

        if not self.images_are_bytes:
            image1_data = image1_data.tobytes()
            image2_data = image2_data.tobytes()

        image1 = Image.open(BytesIO(image1_data)).convert("RGB")
        image2 = Image.open(BytesIO(image2_data)).convert("RGB")

        label = self.labels[index]
        return image1, image2, label
    
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
])

def collate_fn(batch):
    image1_list = []
    image2_list = []
    labels = []
    for img1, img2, label in batch:
        image1_list.append(transform(img1))
        image2_list.append(transform(img2))
        labels.append(label)
    image1_tensor = torch.stack(image1_list) 
    image2_tensor = torch.stack(image2_list) 
    labels_tensor = torch.tensor(labels)    
    return image1_tensor, image2_tensor, labels_tensor

def get_data_loader(path, images_are_bytes = False):
    dataset = PairDataset(path, images_are_bytes)
    dataloader = DataLoader(
		dataset=dataset,
		batch_size=32,         
		shuffle=False,        
		num_workers=2,       
		collate_fn=collate_fn
	)
    return dataloader

def evaluate(
    embeddings1,
    embeddings2,
    issame_list,
    n_folds=10,
    far_target=1e-3,
    n_thresholds=4001,
):
    embeddings1 = np.asarray(embeddings1, dtype=np.float32)
    embeddings2 = np.asarray(embeddings2, dtype=np.float32)
    issame_list = np.asarray(issame_list, dtype=bool)

    embeddings1 = embeddings1.reshape(len(issame_list), -1)
    embeddings2 = embeddings2.reshape(len(issame_list), -1)

    if len(embeddings1) != len(embeddings2) or len(embeddings1) != len(issame_list):
        raise ValueError("The number of elements in embeddings1, embeddings2, and issame_list must be the same.")

    if len(issame_list) < n_folds:
        raise ValueError("The number of pairs must be at least equal to n_folds.")

    # Normalize embeddings before computing cosine similarity.
    embeddings1 = embeddings1 / np.maximum(
        np.linalg.norm(embeddings1, axis=1, keepdims=True), 1e-12
    )
    embeddings2 = embeddings2 / np.maximum(
        np.linalg.norm(embeddings2, axis=1, keepdims=True), 1e-12
    )

    # Higher cosine similarity means the faces are more likely to match.
    similarities = np.sum(embeddings1 * embeddings2, axis=1)

    thresholds = np.linspace(-1.0, 1.0, n_thresholds)
    k_fold = KFold(n_splits=n_folds, shuffle=False)

    tprs = []
    fprs = []
    accuracies = []
    vals = []
    fars = []

    for train_idx, test_idx in k_fold.split(similarities):
        train_sim = similarities[train_idx]
        test_sim = similarities[test_idx]
        train_same = issame_list[train_idx]
        test_same = issame_list[test_idx]

        # Find the best threshold using training pairs only.
        train_acc = np.array([
            np.mean((train_sim >= threshold) == train_same)
            for threshold in thresholds
        ])
        best_threshold = thresholds[np.argmax(train_acc)]

        # Evaluate TPR and FPR on held-out pairs at every threshold.
        fold_tpr = []
        fold_fpr = []

        for threshold in thresholds:
            predict_same = test_sim >= threshold

            tp = np.sum(predict_same & test_same)
            fp = np.sum(predict_same & ~test_same)
            fn = np.sum(~predict_same & test_same)
            tn = np.sum(~predict_same & ~test_same)

            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)

            fold_tpr.append(tpr)
            fold_fpr.append(fpr)

        tprs.append(fold_tpr)
        fprs.append(fold_fpr)

        # Verification accuracy on held-out pairs.
        predict_same = test_sim >= best_threshold
        accuracies.append(np.mean(predict_same == test_same))

        # Select a threshold on training data for the target FAR.
        train_fpr = []
        train_tpr = []

        for threshold in thresholds:
            predict_same = train_sim >= threshold

            negatives = ~train_same
            positives = train_same

            fpr = np.sum(predict_same & negatives) / max(np.sum(negatives), 1)
            tpr = np.sum(predict_same & positives) / max(np.sum(positives), 1)

            train_fpr.append(fpr)
            train_tpr.append(tpr)

        train_fpr = np.asarray(train_fpr)
        train_tpr = np.asarray(train_tpr)

        valid_thresholds = np.where(train_fpr <= far_target)[0]

        if len(valid_thresholds) == 0:
            val_threshold = thresholds[-1]
        else:
            # Maximize TPR while keeping training FAR <= target.
            best_idx = valid_thresholds[
                np.argmax(train_tpr[valid_thresholds])
            ]
            val_threshold = thresholds[best_idx]

        # Measure VAL and FAR using held-out pairs.
        predict_same = test_sim >= val_threshold
        positives = test_same
        negatives = ~test_same

        val = np.sum(predict_same & positives) / max(np.sum(positives), 1)
        far = np.sum(predict_same & negatives) / max(np.sum(negatives), 1)

        vals.append(val)
        fars.append(far)

    tpr = np.mean(tprs, axis=0)
    fpr = np.mean(fprs, axis=0)
    accuracy = np.asarray(accuracies)
    val = float(np.mean(vals))
    val_std = float(np.std(vals))
    far = float(np.mean(fars))

    return tpr, fpr, accuracy, val, val_std, far

def get_embeddings(loader, backbone):
    embeddings1 = []
    embeddings2 = []
    issame_list = []
    
    for i, (image1_tensor, image2_tensor, labels_tensor) in enumerate(loader):
        image1_tensor = image1_tensor.to("cuda")
        image2_tensor = image2_tensor.to("cuda")
    
        with torch.no_grad():
            embedding1 = backbone(image1_tensor)
            embedding2 = backbone(image2_tensor)
    
        embeddings1.append(embedding1.cpu().numpy())
        embeddings2.append(embedding2.cpu().numpy())
        issame_list.extend(labels_tensor.cpu().numpy().tolist())
        
        print(f"Processed batch {i + 1} / {len(loader)}")
        
    embeddings1 = np.concatenate(embeddings1, axis=0)
    embeddings2 = np.concatenate(embeddings2, axis=0)        
    return embeddings1, embeddings2, issame_list
            