#https://github.com/opencv/opencv/issues/20428

#######################################
### Generate ONNX model of vgg16_netvlad
#######################################
import torch
from torchvision import transforms
from PIL import Image
import os


onnx_model_file="./vgg16_netvlad_openibl.onnx"
onnx_model_sim_file="./vgg16_netvlad_openibl_sim.onnx"


# load the best model with PCA (trained by our SFRS)
model = torch.hub.load('yxgeee/OpenIBL', 'vgg16_netvlad', pretrained=True).eval()

# read image
img = Image.open('db1.jpg').convert('RGB') # modify the image path according to your need
transformer = transforms.Compose([transforms.Resize((480, 640)), # (height, width)
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.48501960784313836, 0.4579568627450961, 0.4076039215686255],
                                                       std=[0.00392156862745098, 0.00392156862745098, 0.00392156862745098])])
img = transformer(img)

# use GPU (optional)
#model = model.cuda()
#img = img.cuda()

# extract descriptor (4096-dim)
with torch.no_grad():
    des = model(img.unsqueeze(0))[0]
des = des.cpu().numpy()
print(des)

#print(model)

if not os.path.exists(onnx_model_file):
    torch_onnx_out= torch.onnx.export(model, img.unsqueeze(0), onnx_model_file,
                        export_params=True,
                        verbose=True,
                        input_names=['input'],
                        output_names=["output"],
                        opset_version=11)
else:
    print("ONNX model already exist, skipping export")

#######################################
### Simplify
#######################################
if not os.path.exists(onnx_model_sim_file):
    import onnx
    from onnxsim import simplify

    # load your predefined ONNX model
    model = onnx.load(onnx_model_file)

    # convert model
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnx_model_sim_file)
else:
    print("ONNX simplified model already exist, skipping simplification")

#######################################
### Let's check the correctness of the onnx models
#######################################
import onnxruntime as rt
import numpy as np

def run_onnxruntime(onnx_model, input_tensor, pytorch_ref_output):
    so = rt.SessionOptions()
    so.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_onnx = rt.InferenceSession(onnx_model, sess_options=so)

    input_name = session_onnx.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = session_onnx.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = session_onnx.get_inputs()[0].type
    print("Input type  :", input_type)

    output_name = session_onnx.get_outputs()[0].name
    print("Output name  :", output_name)
    output_shape = session_onnx.get_outputs()[0].shape
    print("Output shape :", output_shape)
    output_type = session_onnx.get_outputs()[0].type
    print("Output type  :", output_type)

    result = session_onnx.run([output_name], {input_name: input_tensor})

    print("Diff between Pytorch and ONNX ({}) : {}".format(onnx_model, np.linalg.norm(result-pytorch_ref_output)))


run_onnxruntime(onnx_model_file, img.unsqueeze(0).cpu().numpy(), des)
run_onnxruntime(onnx_model_sim_file, img.unsqueeze(0).cpu().numpy(), des)
