from torch.utils.tensorboard import SummaryWriter
import platform
import psutil
import torch

def main():
    import shutil

    import MainPredict as MP
    from LabelingTools import Label, Labels

    # labels = Labels(
    #    (
    #        Label("Building", "#3C1098"),
    #        Label("Land", "#8429F6"),
    #        Label("Road", "#6EC1E4"),
    #        Label("Vegetation", "#FEDD3A"),
    #        Label("Water", "#E2A929"),
    #        Label("Unlabeled", "#9B9B9B"),
    #    )
    # )
    labels = Labels(())
    labels.ReadJSON("Data/labels.json")


    pred = MP.Predictor("Data\\kutya\\images", "out", classNum=2, patchSize=640, secondaryPatchSize=64,  IsOutputPOI=True)



    # pred.TrainSemanticModel("Data/kutya", "tmp", labels.list, 100,
    #                         epochs=100,
    #                         model_name=f"foldiKutya_100epoch_attRes_new_cut_CUDA",
    #                         focal_loss_strength=1,  # fix, DO NOT change
    #                         marker_size=16,  # should NEVER affect name of the model
    #                         cut_black_img=False,
    #                         use_regular_model=False,
    #                         batch_num=16,
    #                         filter_num=16,
    #                         create_temp_files=True)  # if marker_size changes it must be true for all iteration.
    writer = SummaryWriter(log_dir='logs')  # Initialize the SummaryWriter

    import time
    start_time = time.time()
    model_name=f"foldiKutya_1000epoch_noFilter_yolov9_multipleRes_{25}%BG_adjustedAugmentation_populate"
    epochs=8
    batch_size=64
    model = pred.TrainIdentificationModel("Data/kutya", "tmp", labels=labels.list,
                                        batch=batch_size,
                                        epochs=epochs,
                                        isPopulate=True,
                                        isUseBackGround=True,
                                        model_name=model_name,
                                        create_temp_files=False,
                                        label_balance=0.25)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Execution took {elapsed_time:.2f} seconds")

    # Log execution time
    writer.add_scalar('Execution Time', elapsed_time)

    # Get the number of available CUDA devices
    num_gpus = torch.cuda.device_count()
    
    # Collect GPU details
    gpu_details = "\n".join(
        [f"CUDA Device {i}: {torch.cuda.get_device_name(i)}" for i in range(num_gpus)]
    ) if num_gpus > 0 else "N/A"
    
    pc_specs = f"""
    System: {platform.system()}
    Node: {platform.node()}
    Release: {platform.release()}
    Version: {platform.version()}
    Machine: {platform.machine()}
    Processor: {platform.processor()}
    RAM: {psutil.virtual_memory().total / (1024 ** 3):.2f} GB
    CUDA Available: {torch.cuda.is_available()}
    {gpu_details}
    """
    writer.add_text('PC Specifications', pc_specs)

    # Log important model details
    try:       
        writer.add_text('Model Details', f"Name: {model_name}, Epochs: {epochs}, Batch Size: {batch_size}")
    except AttributeError:
        print("The model does not have the expected attributes.")

    writer.close()  # Close the writer
    #shutil.rmtree('tmp')
    """
    pred.validate_identification("FoldiKutya\\foldiKutya_1000epoch_noFilter_yolov9_multipleRes_0.25%BG_TEST",
        batch=32,
        data_path='Data\\smolIndie',
        input_labels=labels.list,
        model=None,
        needs_splitting=False)
    """
    # pred.predict_identification("FoldiKutya\\foldiKutya_1000epoch_noFilter_yolov9_multipleRes_0.25%BG_TEST",
    #     input_labels=labels.list,
    #     model=None,
    #     data_path='out\images',
    #     needs_splitting=True)







    #pred.Start("models/nadas_23_08_01_50epoch.keras", labels.list)

    '''
    for clip in range(1, 4, 1):
        for gridSize in range(5, 21, 5):
            pred.TrainSemanticModel("Data/kutya", "tmp", labels.list, 100,
                                    epochs=10, model_name=f"foldiKutya_50epoch_CLAHE_focal_loss5_gauss{7}_sigma{0}_clip{clip/2}_gridSize{gridSize}_CUDA",
                                    gausKernel=7, gausSigma=0, clipLimit=clip/2, tileGridSize=gridSize,
                                    focal_loss_strength=0.5,
                                    create_temp_files=True)

                    ##AIM for the best among the poor'''
if __name__ == '__main__':
    
    main()
