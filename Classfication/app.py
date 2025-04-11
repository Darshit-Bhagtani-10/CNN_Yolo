import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

# Disable Streamlit's file watcher for PyTorch modules
os.environ["STREAMLIT_WATCH_PYTHON_MODULES"] = "false"

# CategoryDictionary - Originally from CategoryDictionary.py
class CategoryDictionary:
    CatName = {
        1: 'Vessel',
        2: 'V Label',
        3: 'V Cork',
        4: 'V Parts GENERAL',
        5: 'Ignore',
        6: 'Liquid GENERAL',
        7: 'Liquid Suspension',
        8: 'Foam',
        9: 'Gel',
        10: 'Solid GENERAL',
        11: 'Granular',
        12: 'Powder',
        13: 'Solid Bulk',
        14: 'Vapor',
        15: 'Other Material',
        16: 'Filled'
    }

    CatLiquid = 6
    CatSolid = 10
    CatFilled = 16
    CatVParts = 4
    SolidLabels = {9, 10, 11, 12, 13}
    LiquidLabels = {6, 7, 9}
    FilledLabels = {6, 7, 8, 9, 10, 11, 12, 13, 15}
    PartsLabels = {2, 3, 4}

    CatLossWeight = {
        'Vessel': 1,
        'V Label': 0.5,
        'V Cork': 0.5,
        'V Parts GENERAL': 0.5,
        'Ignore': 0,
        'Liquid GENERAL': 1,
        'Liquid Suspension': 1,
        'Foam': 1,
        'Gel': 1,
        'Solid GENERAL': 1,
        'Granular': 1,
        'Powder': 1,
        'Solid Bulk': 1,
        'Vapor': 1,
        'Other Material': 1,
        'Filled': 1
    }

    CatNum = {
        'Vessel': -1,
        'V Label': -1,
        'V Cork': -1,
        'V Parts GENERAL': -1,
        'Ignore': -1,
        'Liquid GENERAL': -1,
        'Liquid Suspension': -1,
        'Foam': -1,
        'Gel': -1,
        'Solid GENERAL': -1,
        'Granular': -1,
        'Powder': -1,
        'Solid Bulk': -1,
        'Vapor': -1,
        'Other Material': -1,
        'Filled': -1
    }

    @staticmethod
    def NormalizeWeight(SomeWeight=5, MaxWeight=20):
        SumExamples = 0
        for nm in CategoryDictionary.CatNum:
            if CategoryDictionary.CatNum[nm] >= 0:
                SumExamples += CategoryDictionary.CatNum[nm]
        for nm in CategoryDictionary.CatNum:
            if CategoryDictionary.CatNum[nm] > 0:
                CategoryDictionary.CatLossWeight[nm] *= np.min([(SumExamples / CategoryDictionary.CatNum[nm]), MaxWeight])


# FCN Net model class for semantic segmentation - Originally from FCN_NetModel.py
class Net(nn.Module):
    def __init__(self, CatDict):
        # Generate standard FCN PSP net for image segmentation with only image as input
        super(Net, self).__init__()
        # Load pretrained Resnet 101 encoder
        self.Encoder = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # Create Pyramid Scene Parsing PSP layer
        self.PSPScales = [1, 1 / 2, 1 / 4, 1 / 8]

        self.PSPLayers = nn.ModuleList()  # Layers for decoder
        for Ps in self.PSPScales:
            self.PSPLayers.append(nn.Sequential(
                nn.Conv2d(2048, 1024, stride=1, kernel_size=3, padding=1, bias=True)))
            
        self.PSPSqueeze = nn.Sequential(
            nn.Conv2d(4096, 512, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, stride=1, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        # Skip connection layers for upsampling
        self.SkipConnections = nn.ModuleList()
        self.SkipConnections.append(nn.Sequential(
            nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()))
        self.SkipConnections.append(nn.Sequential(
            nn.Conv2d(512, 256, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()))
        self.SkipConnections.append(nn.Sequential(
            nn.Conv2d(256, 256, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()))
            
        # Skip squeeze applied to the (concat of upsample+skip connection layers)
        self.SqueezeUpsample = nn.ModuleList()
        self.SqueezeUpsample.append(nn.Sequential(
            nn.Conv2d(1024, 512, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()))
        self.SqueezeUpsample.append(nn.Sequential(
            nn.Conv2d(256 + 512, 256, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()))
        self.SqueezeUpsample.append(nn.Sequential(
            nn.Conv2d(256 + 256, 256, stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()))

        # Final prediction layers
        self.OutLayersList = nn.ModuleList()
        self.OutLayersDict = {}
        for f, nm in enumerate(CatDict):
            self.OutLayersDict[nm] = nn.Conv2d(256, 2, stride=1, kernel_size=3, padding=1, bias=False)
            self.OutLayersList.append(self.OutLayersDict[nm])

    def forward(self, Images, UseGPU=True, TrainMode=True, FreezeBatchNormStatistics=False):
        # Convert image to pytorch and normalize values
        RGBMean = [123.68, 116.779, 103.939]
        RGBStd = [65, 65, 65]
        if TrainMode:
            tp = torch.FloatTensor
        else:
            self.half()
            tp = torch.HalfTensor
            
        InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2, 3).transpose(1, 2).type(tp)
        
        if FreezeBatchNormStatistics:
            self.eval()
            
        # Convert to cuda gpu or CPU
        if UseGPU:
            InpImages = InpImages.cuda()
            self.cuda()
        else:
            self = self.cpu()
            self.float()
            InpImages = InpImages.type(torch.float).cpu()
            
        # Normalize image values
        for i in range(len(RGBMean)):
            InpImages[:, i, :, :] = (InpImages[:, i, :, :] - RGBMean[i]) / RGBStd[i]
            
        x = InpImages
        SkipConFeatures = []  # Store features map of layers used for skip connection
        
        # Run Encoder first layer
        x = self.Encoder.conv1(x)
        x = self.Encoder.bn1(x)
        
        # Run remaining encoder layer
        x = self.Encoder.relu(x)
        x = self.Encoder.maxpool(x)
        x = self.Encoder.layer1(x)
        SkipConFeatures.append(x)
        x = self.Encoder.layer2(x)
        SkipConFeatures.append(x)
        x = self.Encoder.layer3(x)
        SkipConFeatures.append(x)
        x = self.Encoder.layer4(x)
        
        # Run psp Layers
        PSPSize = (x.shape[2], x.shape[3])  # Size of the original features map

        PSPFeatures = []  # Results of various of scaled processing
        for i, PSPLayer in enumerate(self.PSPLayers):  # run PSP layers scale features map to various of sizes apply convolution and concat the results
            # Fix for np.int deprecation - use np.int32 instead
            NewSize = (np.array(PSPSize) * self.PSPScales[i]).astype(np.int32)
            if NewSize[0] < 1: NewSize[0] = 1
            if NewSize[1] < 1: NewSize[1] = 1

            y = nn.functional.interpolate(x, tuple(NewSize), mode='bilinear')
            y = PSPLayer(y)
            y = nn.functional.interpolate(y, PSPSize, mode='bilinear')
            PSPFeatures.append(y)
            
        x = torch.cat(PSPFeatures, dim=1)
        x = self.PSPSqueeze(x)
        
        # Upsample features map and combine with layers from encoder using skip connection
        for i in range(len(self.SkipConnections)):
            sp = (SkipConFeatures[-1 - i].shape[2], SkipConFeatures[-1 - i].shape[3])
            x = nn.functional.interpolate(x, size=sp, mode='bilinear')  # Resize
            x = torch.cat((self.SkipConnections[i](SkipConFeatures[-1 - i]), x), dim=1)
            x = self.SqueezeUpsample[i](x)
            
        # Final prediction
        self.OutProbDict = {}
        self.OutLbDict = {}
        
        # Run prediction for each class as binary mask
        for nm in self.OutLayersDict:
            l = self.OutLayersDict[nm](x)
            l = nn.functional.interpolate(l, size=InpImages.shape[2:4], mode='bilinear')  # Resize to original image size
            Prob = F.softmax(l, dim=1)  # Calculate class probability per pixel
            tt, Labels = l.max(1)  # Find label per pixel
            self.OutProbDict[nm] = Prob
            self.OutLbDict[nm] = Labels

        return self.OutProbDict, self.OutLbDict


# Main Streamlit App - Originally from app.py
@st.cache_resource  # Use cache_resource instead of cache_data for PyTorch models
def load_cnn_model(use_gpu_flag=False):
    trained_model_path = 'model/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch'
    net = Net(CategoryDictionary.CatName)
    if use_gpu_flag and torch.cuda.is_available():
        net.load_state_dict(torch.load(trained_model_path))
    else:
        net.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    return net

def do_predictions(input_image, use_gpu_flag, freeze_batch_norm_statistics_flag, model):
    h, w, d = input_image.shape
    r = np.max([h, w])
    if r > 840:
        fr = 840 / r
        input_image = cv2.resize(input_image, (int(w * fr), int(h * fr)))
    img_to_array = np.expand_dims(input_image, axis=0)
    with torch.no_grad():  # Use torch.no_grad() instead of torch.autograd.no_grad()
        out_prob_dict, out_lb_dict = model.forward(Images=img_to_array, TrainMode=False, 
                                                  UseGPU=use_gpu_flag and torch.cuda.is_available(), 
                                                  FreezeBatchNormStatistics=freeze_batch_norm_statistics_flag)
    return out_prob_dict, out_lb_dict, input_image

def plot_predictions(out_lb_dict, resized_image):
    for category_name in out_lb_dict:
        lb = out_lb_dict[category_name].data.cpu().numpy()[0].astype(np.uint8)
        if lb.mean() < 0.001: continue
        if category_name == 'Ignore': continue
        im_overlay = resized_image.copy()
        im_overlay[:, :, 0][lb == 1] = 255
        im_overlay[:, :, 1][lb == 1] = 0
        im_overlay[:, :, 2][lb == 1] = 255
        final_image = np.concatenate([resized_image, im_overlay], axis=1)
        st.write(category_name)
        st.image(final_image)

def main():
    st.write(
        """ # Detecting, segmenting and classifying materials inside mostly transparent vessels  """
    )

    st.sidebar.info("Inference settings")
    use_gpu = st.sidebar.checkbox('Use GPU')
    if use_gpu and not torch.cuda.is_available():
        st.sidebar.warning("CUDA not available - falling back to CPU")
        use_gpu = False
    
    try:
        model = load_cnn_model(use_gpu)
        freeze_batch_norm_statistics = st.sidebar.checkbox('Freeze Batch Norm Statistics')
        st.sidebar.info("Input")
        uploaded_file = st.sidebar.file_uploader("Upload a JPG image of vessel(s)", type=['jpg'])
        if uploaded_file is not None:
            st.sidebar.text('Input image:')
            st.sidebar.image(uploaded_file, width=288)
            predict_button = st.sidebar.button('Predict')
            if predict_button:
                file_bytes = uploaded_file.getvalue()  # Get the bytes without reading - prevents StopIteration error
                image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                out_prob_dict, out_lb_dict, resized_image = do_predictions(image, use_gpu, freeze_batch_norm_statistics, model)
                plot_predictions(out_lb_dict, resized_image)
    except Exception as e:
        st.error(f"Error during model loading or prediction: {str(e)}")
        st.info("Make sure the model file exists at the path: 'model/TrainedModelWeiht1m_steps_Semantic_TrainedWithLabPicsAndCOCO_AllSets.torch'")

if __name__ == "__main__":
    main()