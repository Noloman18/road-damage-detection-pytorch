import os
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from pycocotools.coco import COCO
from engine import train_one_epoch, evaluate
import utils


class CocoDataset(torch.utils.data.Dataset):

    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.image_ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)

        my_annotation = {}
        file_name = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, file_name))

        if len(ann_ids) > 0:
            coco_annotations = coco.loadAnns(ann_ids)
            boxes = []
            labels = []
            areas = []
            for coco_annotation in coco_annotations:
                labels.append(coco_annotation['category_id'])
                areas.append(coco_annotation['area'])
                boxes.append([
                    coco_annotation['bbox'][0],
                    coco_annotation['bbox'][1],
                    coco_annotation['bbox'][2],
                    coco_annotation['bbox'][3]
                ])

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            is_crowd = torch.zeros(len(ann_ids), dtype=torch.int64)
            img_id = torch.as_tensor(img_id, dtype=torch.int64)

            my_annotation['boxes'] = boxes
            my_annotation['labels'] = labels
            my_annotation['image_id'] = img_id
            my_annotation['area'] = areas;
            my_annotation['iscrowd'] = is_crowd

            if self.transforms is not None:
                img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.image_ids)


def get_transform():
    custom_transforms = [torchvision.transforms.ToTensor(),
                         torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(custom_transforms)


def get_model_instance_segmentation(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 6

    dataset = CocoDataset(
        root='D:/Masters/2020/maeda300/images',
        annotation='D:/Masters/2020/maeda300/roadDamage.json',
        transforms=get_transform())

    dataset_test = CocoDataset(
        root='D:/Masters/2020/maeda300/images',
        annotation='D:/Masters/2020/maeda300/roadDamage.json',
        transforms=get_transform())

    indices = torch.randperm(len(dataset)).tolist();
    dataset = torch.utils.data.Subset(dataset, indices[:-700])
    dataset_test = torch.utils.data.Subset(dataset_test,indices[-700:])

    data_loader = \
        torch.utils.data.DataLoader(
            dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=utils.collate_fn)

    data_loader_test = \
        torch.utils.data.DataLoader(
            dataset_test,batch_size=1,shuffle=False,num_workers=0,collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)

    lr_scheduler = \
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=1,
            gamma=0.1)

    num_epochs = 1

    for epoch in range(num_epochs):
        train_one_epoch(model,optimizer,data_loader,device,epoch,print_freq=10)
        lr_scheduler.step()
        evaluate(model,data_loader_test,device=device)

    torch.save(model.state_dict(), 'D:/Masters/2020/maeda300/roadDamage.pkl')
    print("That's it!!!")


main()
