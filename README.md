# 🖐️ Detector de Memes com OpenCV e MediaPipe

Um projeto divertido que utiliza **visão computacional** para detectar gestos com as mãos e exibir imagens de acordo com o gesto reconhecido.
Baseado em **Python**, com as bibliotecas **OpenCV** e **MediaPipe**.

---

## ⚙️ Requisitos

* **Linux** (testado no Ubuntu)
* **Python 3.10.12**
* **OpenCV**
* **MediaPipe**

> ⚠️ O MediaPipe possui limitações quanto à versão do Python.
> Verifique no site oficial quais versões são compatíveis.
> A versão usada neste projeto é **3.10.12**.

---

## 🧩 Instalação passo a passo (Linux)

### 1️⃣ Instalar o `pyenv`

O `pyenv` permite gerenciar múltiplas versões do Python no sistema.

```bash
curl -fsSL https://pyenv.run | bash
```

Adicione as variáveis de ambiente ao seu `~/.bashrc`:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
source ~/.bashrc
```

---

### 2️⃣ Instalar e configurar a versão correta do Python

```bash
pyenv install 3.10.12
cd <pasta-do-projeto>
pyenv local 3.10.12
```

---

### 3️⃣ Criar e ativar um ambiente virtual

```bash
python -m venv .
source bin/activate
```

---

### 4️⃣ Instalar as dependências

```bash
pip install opencv-python mediapipe
```

---

## ▶️ Executar o projeto

Dentro do ambiente virtual, basta rodar:

```bash
python main.py
```

A câmera será ativada automaticamente, e o sistema começará a detectar os gestos das mãos.

---

## 📚 Tecnologias Utilizadas

* [Python 3.10.12](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [MediaPipe](https://developers.google.com/mediapipe)

# PROJETO QUE FOI USADO DE BASE: [NuMetal Detector](https://github.com/GabrielaMarculino/Nu-Metal-Pose-Random-Image-Detector)