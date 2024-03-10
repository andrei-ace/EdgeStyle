FROM dustynv/l4t-ml:r36.2.0

WORKDIR /data/EdgeStyle

COPY efficientvit efficientvit
COPY model model
COPY extract_dataset.py extract_dataset.py
COPY loaddeps.py loaddeps.py
COPY app.py app.py
COPY requirements-jetson-docker.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN python3 loaddeps.py
COPY models/EdgeStyle/controlnet models/EdgeStyle/controlnet
COPY models/clip-vit-large-patch14 models/clip-vit-large-patch14
COPY models/Realistic_Vision_V5.1_noVAE models/Realistic_Vision_V5.1_noVAE
COPY models/sd-vae-ft-mse models/sd-vae-ft-mse
COPY models/control_v11p_sd15_openpose models/control_v11p_sd15_openpose

EXPOSE 7860
ENTRYPOINT ["python3", "app.py"]