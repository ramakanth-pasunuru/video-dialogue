# Train the TriDAF Model
CUDA_VISIBLE_DEVICES=0 python train_retrieval.py --model_name 'model-tridaf' --data_path 'data/fifa' --word_level --use_glove --loss_function '3triplet' --model_save_path 'path_to_model_save_folder' --summary_save_path 'path_to_summary_save_folder' --eval_save_path 'path_to_result_save_folder'



# Test the TriDAF Trained Model
CUDA_VISIBLE_DEVICES=0 python test_retrieval.py --model_name 'name_of_checkpoint' --data_path '/data/fifa' --word_level --model_save_path 'path_to_model_save_folder' --summary_save_path 'path_to_summary_save_folder' --eval_save_path 'path_to_result_save_folder'
