# Neural Network
Creating a simple neural network from scratch using python 

## Try it in the browser

An interactive playground lives in `docs/`. Draw a doodle and a from-scratch CNN
(Conv3x3 -> ReLU -> MaxPool2 x2 -> Dense -> ReLU -> Dense -> Softmax) classifies
it into 345 Quick, Draw! categories, shows the top 3 guesses, and visualises the
network activations as they flow through each layer.

The trained NumPy weights (`main/model_cnn.npz`) are exported to static files
(`docs/model.bin` + `docs/model.json`) and the whole forward pass runs in
JavaScript, so no backend is needed.

Regenerate the web model after retraining:

```powershell
python main/export_web.py
```

Preview locally (must be served over http, not opened as a file):

```powershell
python -m http.server 8000
# then open http://localhost:8000/docs/
```

## Deploying to GitHub Pages

1. Commit and push the `docs/` folder (including `model.bin` and `model.json`).
2. In the repo: **Settings -> Pages**.
3. Under **Build and deployment**, set **Source** to *Deploy from a branch*,
   choose branch **main** and folder **/docs**, then save.

The site will be published at `https://<user>.github.io/<repo>/`.
