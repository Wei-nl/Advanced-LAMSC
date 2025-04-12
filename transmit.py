import training_process

if __name__ == '__main__':
    dataset = "example"

sem_aware_images_path = "data_voc2012/semantic_feature_maps/"
training_process.data_transmission(sem_aware_images_path,dataset=dataset)