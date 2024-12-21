from pytorch_lightning import Trainer
from models.BaseModel import BaseModel
from dataloader.FireSpreadDataModule import FireSpreadDataModule

def main():
    # Initialize model
    model = BaseModel(
        n_channels=40,
        flatten_temporal_dimension=True,
        pos_class_weight=2.0,
        loss_function="BCE",
        use_doy=False,
        required_img_size=(128, 128)
    )
    
    # Initialize datamodule
    datamodule = FireSpreadDataModule(
        data_dir='/workspace/dataset/processed',
        batch_size=16,
        n_leading_observations=5,
        n_leading_observations_test_adjustment=5,
        crop_side_length=128,
        load_from_hdf5=True,
        num_workers=4,
        remove_duplicate_features=False
    )
    datamodule.setup(stage="fit")

    # Initialize trainer
    trainer = Trainer(max_epochs=10, accelerator="gpu", devices=1)
    
    # Train model
    trainer.fit(model, datamodule)

if __name__ == "__main__":
    main()
