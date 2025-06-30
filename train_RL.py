import os
import torch
import random
import tempfile
import trimesh
import open3d
import skimage
import numpy as np
from tqdm import tqdm
from cadrille import Cadrille, collate
from transformers import AutoTokenizer, AutoProcessor
from torchvision import transforms
from PIL import Image, ImageOps
from pathlib import Path
import cadquery as cq

from utils import code_to_image
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Загрузка модели и процессора ===
model = Cadrille.from_pretrained('Qwen/Qwen2-VL-2B-Instruct').to(device)
model.rope_deltas = None
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)



# === Получение всех train подпапок ===
train_root = Path('./cad-recode/train/')
subsets = sorted([p for p in train_root.iterdir() if p.is_dir()])
val_root = Path('./cad-recode/val/')

# === Пустой код ===
empty_code = "import cadquery as cq\nresult = cq.Workplane()"
empty_image = code_to_image(empty_code)

# === Основной цикл обучения ===
N_epochs = 3
N_points = 64  # для point cloud
for epoch in range(N_epochs):
    for i_ss, subset_dir in enumerate(subsets):
        py_files = list(subset_dir.glob("*.py"))
        random.shuffle(py_files)

        for py_file in tqdm(py_files, desc=f"[Subset {i_ss}] Epoch {epoch}"):
            with open(py_file, 'r', encoding='utf-8') as f:
                script = f.read()

            target_image = code_to_image(script)
            cur_code = empty_code
            cur_image = empty_image

            xs = []
            for step in range(i_ss + 1):
                sample = {
                    'video': [target_image, cur_image],  # двукадровое "видео"
                    'description': cur_code,
                    'answer': script
                }
                batch = [sample]

                # === Подготовка входов для модели ===
                inputs = collate(batch, processor, n_points=N_points, eval=False)
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

                model.train()
                outputs = model(**inputs)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # === Генерация нового кода ===
                model.eval()
                with torch.no_grad():
                    gen_inputs = collate(batch, processor, n_points=N_points, eval=True)
                    gen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in gen_inputs.items()}
                    # Заглушка. Удаляем, потому что нужно только для логгирования и выводит ошибку
                    if 'file_name' in gen_inputs:
                        del gen_inputs['file_name']
                    generated_ids = model.generate(
                        **gen_inputs,
                        max_new_tokens=512,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.8
                    )
                    cur_code = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    cur_image = code_to_image(cur_code)
                    print('generated')
