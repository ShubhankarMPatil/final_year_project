import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import cv2
import numpy

# setup device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)

def findCaption(image):
    # load image
    raw_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # preprocess the image
    # vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    # generate caption
    outputs = model.generate({"image": image})
    # ['a large fountain spewing water into the air']
    return outputs

def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

    
        result = findCaption(frame)
        print(result)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()