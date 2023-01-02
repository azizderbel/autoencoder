import torch
import torchinfo
from model_structure import AE
from model_train import train
import utils as utils
import matplotlib.pyplot as plt

model = AE()
#train(model=model,epochs=100)
latent_sapce = utils.create_auto_encoder_latent_space(visualize_latent_space=True,train_classifier=False)
utils.predict_image_class(image = utils.get_image()[0],
                            auto_encoder = utils.load_saved_auto_encoder(),
                            classifier = utils.load_saved_classifier(),
                            visualisation = True
                          )
mapped_latent_space = utils.map_latent_space_probability_space(latent_space=latent_sapce,
                                            auto_encoder=utils.load_saved_auto_encoder(),
                                            classifier = utils.load_saved_classifier(),
                                            )
utils.visualize_mapping(mapped_latent_space=mapped_latent_space,save=True)
plt.show()


