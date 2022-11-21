from .base_dataset import BaseDataset
import torch

class VQACPDataset(BaseDataset):
    def __init__(self, *args, split="", draw_false_image_vqa=0, **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        self.draw_false_image_vqa = draw_false_image_vqa

        if split == "train":
            names = ["vqav2_cp_train"]
        elif split == "val":
            names = ["vqav2_cp_val"]
        elif split == "test":
            names = ["vqav2_cp_test"]

        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        if getattr(self, 'table', None) is None:
            self.open_arrow()

        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        if self.split != "test":
            answers = self.table["answers"][index][question_index].as_py()
            labels = self.table["answer_labels"][index][question_index].as_py()
            scores = self.table["answer_scores"][index][question_index].as_py()
        else:
            answers = list()
            labels = list()
            scores = list()

        data = {"image": image_tensor,
               "text": text,
               "vqa_answer": answers,
               "vqa_labels": labels,
               "vqa_scores": scores,
               "qid": qid}

        if self.draw_false_image_vqa > 0:
            false_image = []
            for i in range(self.draw_false_image_vqa):
                false_image_tensor = self.get_false_image(f"vqa")
                false_image.append(false_image_tensor['false_image_vqa'][0].unsqueeze(0))
            false_image = torch.cat(false_image, 0)
            data.update({"false_image": false_image, 
                         "yes_type": int('yes' in answers or 'no' in answers)})

        return data
