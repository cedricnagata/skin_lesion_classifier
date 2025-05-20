import modal

stub = modal.Stub("skin-lesion-api")

image = (
    modal.Image.debian_slim()
    .pip_install("flask", "tensorflow", "pillow", "numpy")
    .add_local_file("skin_lesion_classifier_85.keras")
    .add_local_file("app.py")
)

@stub.function(image=image, ports=[8000])
def run():
    import subprocess
    subprocess.run(["python", "app.py"])