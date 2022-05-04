from .base_training_loop import GANTrainingLoop


#-------------------------------------------------------------------------

class TrainingPix2Pix(GANTrainingLoop):

    def __init__(self,
                 Model: object,
                 dataset: object,
                 train_generator: object,
                 val_generator: object,
                 config: dict):
        super().__init__(Model,
                         dataset,
                         train_generator,
                         val_generator,
                         config)

    def train(self, verbose=1):

        """ Main training loop for Pix2Pix """

        self.results = {}
        self.results["g_metric"] = []
        self.results["d_metric"] = []
        self.results["train_L1"] = []
        self.results["val_L1"] = []
        self.results["epochs"] = []
        self.results["time"] = 0

        super().train(verbose)
    
    def _process_and_save_images(self, epoch, phase="validation", tuning_path=None):

        """ Saves sample of images """

        if phase == "train":
            data_generator = self.train_generator
        
        elif phase == "validation":
            data_generator = self.val_generator

        data = data_generator.example_images()

        if "target_times" in data.keys():
            pred = self.Model.Generator(data["real_source"], data["target_times"]).numpy()
        else:
            pred = self.Model.Generator(data["real_source"]).numpy()

        source = data_generator.un_normalise(data["real_source"])
        target = data_generator.un_normalise(data["real_target"])
        pred = data_generator.un_normalise(pred)

        self._save_images(epoch, phase, tuning_path, source, target, pred)

    def _save_model(self):
        self.Model.Generator.save_weights(f"{self.MODEL_SAVE_PATH}/generator.ckpt")
