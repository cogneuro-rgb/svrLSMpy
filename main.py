from pathlib import Path
from modules.run_svr_lsm_iteration import run_svr_lsm_iteration

if __name__ == "__main__":
    base_folder = Path.cwd()  # CURRENT DIRECTORY
    base_folder = base_folder / "symptoms"



    '''
    symptom_folder = base_folder / 'Dysmetropsia'
    csv_name = "Dysmetropsia.csv"

    param_grid = {
        'C': [50,40,30, 20, 10, 5],
        'gamma': [10, 5, 4, 3, 2, 1],
        'epsilon': [0.1]
    }
    param_grid = {
        'C': [50],
        'gamma': [1],
        'epsilon': [0.1]
    }
    behaviour_name = "Dysmetropsia"
    run_svr_lsm_iteration(symptom_folder=symptom_folder,
                          csv_name=csv_name,
                          behaviour_name=behaviour_name,
                          normalize_vector=True,
                          do_regress_out_lesion_volume=True,
                          max_score=100,
                          min_patient_count=5,
                          param_grid=param_grid,
                          n_permutations=1,
                          alpha=0.05,
                          n_splits=5,
                          num_slices=7)





    '''


    symptom_folder = base_folder / 'VAST'
    csv_name = "VAST_Data_Fluency.csv"

    param_grid = {
        'C': [50, 40, 30, 20, 10, 5],
        'gamma': [10, 5, 4, 3, 2, 1],
        'epsilon': [0.1]
    }

    param_grid = {
        'C': [50],
        'gamma': [2],
        'epsilon': [0.1]
    }
    behaviour_name = "Aphasia"
    run_svr_lsm_iteration(symptom_folder=symptom_folder,
                          csv_name=csv_name,
                          behaviour_name=behaviour_name,
                          normalize_vector=True,
                          do_regress_out_lesion_volume=True,
                          max_score=4,
                          min_patient_count="10%", #number or string with or without %
                          param_grid=param_grid,
                          n_permutations=1,
                          alpha=0.05,
                          n_splits=2,
                          num_slices=7)
    '''
    # DYNAMIC EMOTION
    symptom_folder = base_folder / 'Dynamic_Emotion'
    csv_name = "Dynamic_Emotion_scores.csv"

    param_grid = {
        'C': [50, 40, 30, 20, 10, 5],
        'gamma': [10, 5, 4, 3, 2, 1],
        'epsilon': [0.1]
    }

    behaviour_name = "Dynamic Emotion Recognition"
    run_svr_lsm_iteration(symptom_folder=symptom_folder,
                          csv_name=csv_name,
                          behaviour_name=behaviour_name,
                          normalize_vector=True,
                          do_regress_out_lesion_volume=True,
                          max_score=100,
                          min_patient_count=10,
                          param_grid=param_grid,
                          n_permutations=1000,
                          alpha=0.05,
                          n_splits=5,
                          num_slices=7)
    '''
